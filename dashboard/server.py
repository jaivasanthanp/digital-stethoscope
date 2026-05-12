"""
Local Digital Stethoscope dashboard (CirCor 2022 / PhysioNet build).

Laptop role:
    - Accept a WAV upload (any length) or a 1D audio .npy
    - Resample to 4 kHz mono, segment into successive 2-second windows
    - Stream each window's int16 PCM over UART to the STM32
    - Display the STM32-computed mel spectrogram and classification

STM32 role:
    - Receive 8000 int16 LE samples through the 'A' upload protocol
    - Compute audio RMS/peak/ZCR + log-mel spectrogram on-chip
    - Run the INT8 ResNet-10 + calibrated unknown gate locally
    - Return class + raw probabilities + per-stage latency + spectrogram

Usage:
    py dashboard\\server.py --port COM6 --baud 115200
    http://127.0.0.1:8765

Optional second file upload: a labels JSON (e.g. mixed_demo_labels.json) so
the dashboard can show per-segment accuracy against ground truth.
"""

from __future__ import annotations

import argparse
import base64
import cgi
import io
import json
import struct
import tempfile
import threading
import time
import wave
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

try:
    import librosa
except ImportError as exc:
    raise SystemExit("librosa is required. Install ml/requirements.txt first.") from exc

try:
    import serial
    from serial.tools import list_ports
except ImportError as exc:
    raise SystemExit("pyserial is required. Install ml/requirements.txt first.") from exc


ROOT = Path(__file__).resolve().parents[1]
CIRCOR_DIR = ROOT / "ml" / "data_circor"
DEMO_DIR = CIRCOR_DIR / "demo"
PRESENTATION_DIR = ROOT / "presentation_samples"
GENERIC_AUDIO_DIRS = [PRESENTATION_DIR, DEMO_DIR, CIRCOR_DIR / "raw" / "training_data"]

CLASS_NAMES = ["Absent", "Present", "Unknown"]
CLASS_KEYS = ["absent", "present", "unknown"]
CLASS_COLORS = ["#14866d", "#b32424", "#ac6600"]

RESPONSE_MAGIC = 0xA5
UPLOAD_ACK_MAGIC = 0xA6

SR_TARGET = 4000
WIN_SEC = 2.0
WIN_SAMPLES = int(SR_TARGET * WIN_SEC)
HOST_SUBCHUNK_ELEMENTS = 128
N_FFT = 512
HOP_LENGTH = 128
N_MELS = 64
SPEC_SIZE = 64

# Extended STM32 response: 1 magic + 1 class + 1 conf + 3 raw probs
# + 1 gate_applied + 5 u16 LE words + 4096 float32 mel values
RESPONSE_HEADER_BYTES = 1 + 1 + 1 + 3 + 1 + 2 + 2 + 2 + 2 + 2
RESPONSE_SPEC_BYTES = SPEC_SIZE * SPEC_SIZE * 4
RESPONSE_TOTAL_BYTES = RESPONSE_HEADER_BYTES + RESPONSE_SPEC_BYTES


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Digital Stethoscope - CirCor 2022 STM32 ML Dashboard</title>
  <style>
    :root {
      --bg: #f4f6f9;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #4f5b69;
      --line: #d0d6dd;
      --rule: #c8ccd1;
      --accent: #1f4e79;
      --accent-2: #3573a6;
      --ok: #14866d;
      --warn: #ac6600;
      --bad: #b32424;
      --absent: #14866d;
      --present: #b32424;
      --unknown: #ac6600;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Roboto", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 22px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 { margin: 0; font-size: 22px; font-weight: 650; }
    .sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
    .actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .uart-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 14px;
      font-size: 12.5px;
      color: var(--muted);
      background: #fff;
      min-width: 158px;
      justify-content: center;
    }
    .uart-pill::before {
      content: "";
      width: 9px; height: 9px;
      border-radius: 50%;
      background: #98a2b3;
    }
    .uart-pill.connected::before { background: var(--ok); }
    .uart-pill.disconnected::before { background: var(--bad); }
    main {
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
      padding: 14px 18px 36px;
      max-width: 1380px;
      margin: 0 auto;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 14px;
    }
    .label {
      color: var(--ink);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 10px;
    }
    .tabs {
      display: inline-flex;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }
    .tab {
      height: 36px;
      padding: 0 18px;
      border: 0;
      background: #ffffff;
      color: var(--ink);
      cursor: pointer;
      font-weight: 600;
      font-size: 13px;
    }
    .tab.active { background: var(--accent); color: #fff; }
    .controls {
      display: grid;
      grid-template-columns: 1.4fr 1.2fr 140px 110px auto auto;
      gap: 12px;
      align-items: end;
    }
    .controls.loop {
      grid-template-columns: 1.4fr 140px 110px 130px auto auto;
    }
    label.field {
      display: grid;
      gap: 6px;
      font-size: 12.5px;
      color: var(--muted);
    }
    input, select, button {
      height: 38px;
      border-radius: 5px;
      border: 1px solid var(--line);
      font: inherit;
      color: var(--ink);
      background: #fff;
    }
    input[type="file"] { padding: 6px 8px; font-size: 12.5px; }
    input[type="number"], select { padding: 0 10px; }
    button.primary {
      background: var(--accent);
      color: #fff;
      border: 0;
      padding: 0 18px;
      cursor: pointer;
      font-weight: 650;
    }
    button.primary:hover { background: var(--accent-2); }
    button.secondary {
      background: #fff;
      color: var(--ink);
      cursor: pointer;
      padding: 0 14px;
    }
    button:disabled { opacity: 0.55; cursor: wait; }
    .summary-grid {
      display: grid;
      grid-template-columns: 1.4fr repeat(4, 1fr);
      gap: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
      background: #fbfcfd;
      min-height: 96px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 11.5px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .metric strong {
      display: block;
      font-size: 22px;
      font-weight: 700;
      line-height: 1.15;
      overflow-wrap: anywhere;
    }
    .result-card {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px 16px;
      display: grid;
      grid-template-columns: 60px 1fr 110px;
      gap: 14px;
      align-items: center;
      background: #fbfcfd;
    }
    .result-card.absent { background: #ecf7f1; border-color: #82c9aa; }
    .result-card.present { background: #fdecec; border-color: #e8a3a3; }
    .result-card.unknown { background: #fdf5e6; border-color: #e0b677; }
    .pulse-icon {
      width: 54px; height: 54px;
      border-radius: 50%;
      background: rgba(255,255,255,0.8);
      display: grid;
      place-items: center;
      font-size: 24px;
      font-weight: 700;
      animation: pulse 1.45s ease-in-out infinite;
    }
    @keyframes pulse {
      0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(31,78,121,0.22); }
      50% { transform: scale(1.06); box-shadow: 0 0 0 14px rgba(31,78,121,0); }
    }
    .gauge { width: 160px; height: 95px; }
    .gauge text { font: 700 22px Arial; fill: var(--ink); }
    .panel-title {
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 10px;
    }
    .charts {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    canvas {
      display: block;
      width: 100%;
      height: 220px;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 4px;
    }
    canvas.tall { height: 260px; }
    canvas.short { height: 130px; }
    .latency-grid, .stats-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td {
      text-align: left;
      padding: 8px 9px;
      border-bottom: 1px solid var(--rule);
      white-space: nowrap;
    }
    th { font-size: 12px; color: var(--ink); background: #f0f3f7; }
    tr.segment-row { cursor: pointer; }
    tr.segment-row:hover { background: #eef3fa; }
    tr.segment-row.active { background: #e3edf7; font-weight: 600; }
    tr.match-ok td:nth-child(4) { color: var(--ok); font-weight: 700; }
    tr.match-fail td:nth-child(4) { color: var(--bad); font-weight: 700; }
    .pill {
      display: inline-block;
      padding: 1px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    .pill.absent { background: rgba(20,134,109,0.15); color: var(--absent); }
    .pill.present { background: rgba(179,36,36,0.13); color: var(--present); }
    .pill.unknown { background: rgba(172,102,0,0.13); color: var(--unknown); }
    .pill.gate { background: rgba(53,115,166,0.15); color: var(--accent-2); }
    .progress {
      height: 10px;
      background: #e8edf3;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 6px;
    }
    .progress-fill {
      height: 100%; width: 0%;
      background: var(--accent);
      transition: width 220ms ease;
    }
    .timeline-canvas { height: 140px; }
    .panel-head {
      display: flex; align-items: center; justify-content: space-between;
      gap: 10px; margin-bottom: 10px;
    }
    pre.log {
      margin: 0;
      min-height: 78px;
      max-height: 220px;
      overflow: auto;
      white-space: pre-wrap;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 10px;
      font-family: Consolas, ui-monospace, monospace;
      font-size: 12px;
    }
    pre.error { color: var(--bad); }
    .small { color: var(--muted); font-size: 12px; }
    .hidden { display: none !important; }
    @media (max-width: 960px) {
      .controls, .controls.loop { grid-template-columns: 1fr; }
      .summary-grid { grid-template-columns: 1fr; }
      .charts { grid-template-columns: 1fr; }
      .latency-grid, .stats-grid { grid-template-columns: repeat(2, 1fr); }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Digital Stethoscope - STM32 ML Dashboard</h1>
      <div class="sub">PhysioNet/CinC 2022 CirCor labels &middot; on-chip DSP + INT8 ResNet-10 + calibrated unknown gate</div>
    </div>
    <div class="actions">
      <div class="tabs">
        <button id="customTab" class="tab active" type="button">Custom Upload</button>
        <button id="loopTab" class="tab" type="button">Generic Loop</button>
      </div>
      <div id="uartStatus" class="uart-pill">COM Disconnected</div>
    </div>
  </header>
  <main>
    <section id="customPanel">
      <div class="label">Audio Upload</div>
      <form id="classifyForm" class="controls">
        <label class="field">WAV / 1D NPY audio
          <input id="audioFile" name="file" type="file" accept=".wav,.npy,audio/wav,audio/x-wav" required>
        </label>
        <label class="field">Ground-truth labels JSON (optional)
          <input id="labelsFile" name="labels" type="file" accept=".json,application/json">
        </label>
        <label class="field">Serial port
          <select id="serialPort" name="port"></select>
        </label>
        <label class="field">Baud
          <input id="baud" name="baud" type="number" value="115200" min="9600" step="1">
        </label>
        <button id="submitBtn" class="primary" type="submit">Classify</button>
        <button id="clearBtn" class="secondary" type="button">Clear</button>
      </form>
      <div id="progressWrap" class="hidden">
        <div class="small" id="progressText">Processing segment 0 of 0...</div>
        <div class="progress"><div id="progressFill" class="progress-fill"></div></div>
      </div>
    </section>

    <section id="loopPanel" class="hidden">
      <div class="label">Generic Demo Loop</div>
      <div class="controls loop">
        <label class="field">Generic source
          <select id="genericSource"></select>
        </label>
        <label class="field">Serial port
          <select id="loopSerialPort"></select>
        </label>
        <label class="field">Baud
          <input id="loopBaud" type="number" value="115200" min="9600" step="1">
        </label>
        <label class="field">Pause sec
          <input id="loopInterval" type="number" value="3" min="1" step="1">
        </label>
        <button id="startLoopBtn" class="primary" type="button">Start Loop</button>
        <button id="stopLoopBtn" class="secondary" type="button" disabled>Stop</button>
      </div>
    </section>

    <section>
      <div class="label">Aggregate Result</div>
      <div class="summary-grid">
        <div id="resultCard" class="result-card">
          <div id="pulseIcon" class="pulse-icon">~</div>
          <div>
            <div id="sourceName" class="small">-</div>
            <div id="className" style="font-size:24px;font-weight:700;">-</div>
            <div id="resultMeta" class="small">-</div>
          </div>
          <div style="display:grid;justify-items:center;">
            <svg class="gauge" viewBox="0 0 200 110" aria-label="confidence gauge">
              <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#e4e7ec" stroke-width="18" stroke-linecap="round"/>
              <path id="gaugeArc" d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#1f4e79" stroke-width="18" stroke-linecap="round" pathLength="100" stroke-dasharray="0 100"/>
              <text id="confidence" x="100" y="86" text-anchor="middle">-</text>
            </svg>
            <div class="small">Confidence</div>
          </div>
        </div>
        <div class="metric"><span>Segments</span><strong id="metricSegments">-</strong></div>
        <div class="metric"><span>Duration</span><strong id="metricDuration">-</strong></div>
        <div class="metric"><span>Mean DSP+Inference</span><strong id="metricLatency">-</strong></div>
        <div class="metric"><span>Accuracy vs Ground Truth</span><strong id="metricAccuracy">n/a</strong></div>
      </div>
    </section>

    <section>
      <div class="label">Per-Segment Probability Timeline</div>
      <canvas id="probChart" class="tall" width="1200" height="320"></canvas>
      <div class="small">Three coloured traces show the raw model probability (before unknown-gate override) for each 2-second window streamed to the STM32. Gate-overridden windows are marked with a circle.</div>
    </section>

    <section>
      <div class="label">Classification Timeline (color = STM32 verdict)</div>
      <canvas id="timelineCanvas" class="timeline-canvas" width="1200" height="160"></canvas>
      <div class="small" id="timelineHint">Click a block to load that segment's mel spectrogram and audio stats below.</div>
    </section>

    <section class="charts">
      <div>
        <div class="label">Per-Class Cumulative Counts</div>
        <canvas id="classBarChart" width="900" height="240"></canvas>
      </div>
      <div>
        <div class="label">Audio RMS over Time</div>
        <canvas id="rmsChart" width="900" height="240"></canvas>
      </div>
    </section>

    <section>
      <div class="label">Latency Breakdown (mean ms / segment, STM32-reported)</div>
      <div class="latency-grid">
        <div class="metric"><span>STM32 DSP</span><strong id="lat_dsp">-</strong></div>
        <div class="metric"><span>STM32 Inference</span><strong id="lat_infer">-</strong></div>
        <div class="metric"><span>UART Upload (audio)</span><strong id="lat_upload">-</strong></div>
        <div class="metric"><span>UART Return (spec)</span><strong id="lat_return">-</strong></div>
      </div>
    </section>

    <section class="charts">
      <div>
        <div class="panel-head">
          <div class="label" style="margin:0;">Selected-Segment Waveform</div>
          <div id="segmentLabel" class="small">no segment selected</div>
        </div>
        <canvas id="segmentWaveform" width="900" height="220"></canvas>
        <div class="stats-grid" style="margin-top:12px;">
          <div class="metric"><span>RMS (norm)</span><strong id="seg_rms">-</strong></div>
          <div class="metric"><span>Peak (norm)</span><strong id="seg_peak">-</strong></div>
          <div class="metric"><span>Zero-Crossing Rate (Hz)</span><strong id="seg_zcr">-</strong></div>
          <div class="metric"><span>Gate Applied?</span><strong id="seg_gate">-</strong></div>
        </div>
      </div>
      <div>
        <div class="panel-head">
          <div class="label" style="margin:0;">STM32-Computed Mel Spectrogram</div>
          <div class="small">64x64 log-mel, normalised on chip</div>
        </div>
        <canvas id="specCanvas" width="900" height="220"></canvas>
        <div class="small" style="margin-top:8px;">Axes: time bin (left to right, hop = 32 ms) vs mel bin (bottom = 25 Hz, top = 2000 Hz).</div>
      </div>
    </section>

    <section>
      <div class="label">Audio Playback (uploaded file, 4 kHz mono)</div>
      <audio id="audioPlayer" controls style="width:100%;"></audio>
    </section>

    <section>
      <div class="panel-head">
        <div class="label" style="margin:0;">Per-Segment Results</div>
        <button id="exportBtn" class="secondary" type="button">Export CSV</button>
      </div>
      <div style="overflow:auto; max-height:340px; border:1px solid var(--line); border-radius:4px;">
        <table>
          <thead><tr>
            <th>#</th><th>Start</th><th>Class (STM32)</th><th>Ground Truth</th>
            <th>Confidence</th><th>P(Absent)</th><th>P(Present)</th><th>P(Unknown)</th>
            <th>DSP ms</th><th>Infer ms</th><th>RMS</th>
          </tr></thead>
          <tbody id="segmentTable"></tbody>
        </table>
      </div>
    </section>

    <section>
      <div class="label">Status / Log</div>
      <pre id="log" class="log">Ready.</pre>
    </section>

    <section>
      <details>
        <summary style="cursor:pointer;color:var(--muted);font-size:12px;letter-spacing:0.04em;text-transform:uppercase;">Model &amp; Build Info</summary>
        <div class="latency-grid" style="margin-top:12px;">
          <div class="metric"><span>Architecture</span><strong>ResNet-10 INT8</strong></div>
          <div class="metric"><span>Runtime</span><strong>TFLite Micro + CMSIS-NN</strong></div>
          <div class="metric"><span>Model size</span><strong>102.3 KB</strong></div>
          <div class="metric"><span>Dataset</span><strong>CirCor 2022</strong></div>
          <div class="metric"><span>Classes</span><strong>Absent / Present / Unknown</strong></div>
          <div class="metric"><span>Test accuracy (w/ gate)</span><strong>73.2%</strong></div>
          <div class="metric"><span>Target</span><strong>STM32U575 @ 160 MHz</strong></div>
          <div class="metric"><span>On-chip DSP latency</span><strong>~30 ms / window</strong></div>
        </div>
        <div class="small" style="margin-top:10px;">
          The unknown gate elevates argmax results to "Unknown" when the winning probability is small or the top-2 margin is tight. Calibrated on the CirCor validation split.
        </div>
      </details>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const CLASS_NAMES = ["Absent", "Present", "Unknown"];
    const CLASS_KEYS  = ["absent", "present", "unknown"];
    const CLASS_COLORS = ["#14866d", "#b32424", "#ac6600"];
    const CLASS_ICONS = ["A", "M", "?"];

    let lastPayload = null;
    let selectedSegment = 0;
    let loopTimer = null;
    let loopBusy = false;
    let loopNext = 0;

    function log(msg, isError = false) {
      $("log").textContent = msg;
      $("log").className = isError ? "log error" : "log";
    }

    function setMode(mode) {
      const custom = mode === "custom";
      $("customTab").classList.toggle("active", custom);
      $("loopTab").classList.toggle("active", !custom);
      $("customPanel").classList.toggle("hidden", !custom);
      $("loopPanel").classList.toggle("hidden", custom);
    }

    function fmtMs(v) {
      if (v === null || v === undefined || isNaN(Number(v))) return "-";
      return `${Math.round(Number(v))} ms`;
    }

    function fmtPct(v) {
      if (v === null || v === undefined || isNaN(Number(v))) return "-";
      return `${Math.round(Number(v))}%`;
    }

    function setProgress(done, total) {
      const wrap = $("progressWrap");
      if (!total) {
        wrap.classList.add("hidden");
        return;
      }
      wrap.classList.remove("hidden");
      const pct = Math.round((done / total) * 100);
      $("progressFill").style.width = `${pct}%`;
      $("progressText").textContent = `Processing segment ${done} of ${total} (${pct}%)`;
    }

    function classKey(classId) {
      return (classId >= 0 && classId < CLASS_KEYS.length) ? CLASS_KEYS[classId] : "unknown";
    }

    function renderAggregate(payload) {
      const agg = payload.aggregate || {};
      const dominantClass = agg.dominant_class_id ?? 0;
      const ckey = classKey(dominantClass);
      $("resultCard").className = `result-card ${ckey}`;
      $("pulseIcon").textContent = CLASS_ICONS[dominantClass] || "~";
      $("sourceName").textContent = payload.file_name || "-";
      $("className").textContent = CLASS_NAMES[dominantClass] || "-";
      const dominantCount = agg.counts ? (agg.counts[CLASS_KEYS[dominantClass]] || 0) : 0;
      $("resultMeta").textContent = `${dominantCount} / ${payload.n_segments} segments &middot; ${(payload.audio_seconds || 0).toFixed(1)} s window`;
      $("metricSegments").textContent = payload.n_segments;
      $("metricDuration").textContent = `${(payload.audio_seconds || 0).toFixed(1)} s`;
      $("metricLatency").textContent = fmtMs((agg.avg_dsp_ms || 0) + (agg.avg_inference_ms || 0));
      $("metricAccuracy").textContent = (agg.accuracy_pct === null || agg.accuracy_pct === undefined)
        ? "n/a" : `${agg.accuracy_pct.toFixed(1)}%`;
      $("confidence").textContent = `${Math.round(agg.dominant_confidence || 0)}%`;
      $("gaugeArc").setAttribute("stroke-dasharray", `${Math.max(0, Math.min(100, agg.dominant_confidence || 0))} 100`);
      $("gaugeArc").setAttribute("stroke", CLASS_COLORS[dominantClass] || "#1f4e79");

      $("lat_dsp").textContent = fmtMs(agg.avg_dsp_ms);
      $("lat_infer").textContent = fmtMs(agg.avg_inference_ms);
      $("lat_upload").textContent = fmtMs(agg.avg_uart_upload_ms);
      $("lat_return").textContent = fmtMs(agg.avg_uart_return_ms);
    }

    function renderSegments(payload) {
      const tbody = $("segmentTable");
      tbody.innerHTML = "";
      payload.segments.forEach((seg, idx) => {
        const tr = document.createElement("tr");
        tr.className = `segment-row ${idx === selectedSegment ? "active" : ""}`;
        const probs = seg.raw_probabilities || [0, 0, 0];
        if (seg.ground_truth_class_id !== null && seg.ground_truth_class_id !== undefined) {
          tr.classList.add(seg.class_id === seg.ground_truth_class_id ? "match-ok" : "match-fail");
        }
        tr.innerHTML = `
          <td>${idx + 1}</td>
          <td>${seg.start_seconds.toFixed(1)} s</td>
          <td><span class="pill ${CLASS_KEYS[seg.class_id]}">${seg.class_name}</span>
              ${seg.gate_applied ? '<span class="pill gate" title="Unknown gate fired">gated</span>' : ''}</td>
          <td>${seg.ground_truth_class_name || '-'}</td>
          <td>${seg.confidence}%</td>
          <td>${probs[0]}%</td>
          <td>${probs[1]}%</td>
          <td>${probs[2]}%</td>
          <td>${seg.latency.stm32_dsp_ms}</td>
          <td>${seg.latency.stm32_inference_ms}</td>
          <td>${(seg.audio_stats.rms).toFixed(3)}</td>
        `;
        tr.addEventListener("click", () => selectSegment(idx));
        tbody.appendChild(tr);
      });
    }

    function selectSegment(idx) {
      if (!lastPayload || !lastPayload.segments[idx]) return;
      selectedSegment = idx;
      const seg = lastPayload.segments[idx];
      $("segmentLabel").textContent = `Segment ${idx + 1} at ${seg.start_seconds.toFixed(1)}-${seg.end_seconds.toFixed(1)} s`;
      $("seg_rms").textContent = seg.audio_stats.rms.toFixed(3);
      $("seg_peak").textContent = seg.audio_stats.peak.toFixed(3);
      $("seg_zcr").textContent = Math.round(seg.audio_stats.zcr / 2); // rough Hz
      $("seg_gate").textContent = seg.gate_applied ? "yes" : "no";
      drawSegmentWaveform(seg.waveform_preview);
      drawSpectrogram(seg.spectrogram);
      renderSegments(lastPayload);
      drawTimeline(lastPayload);
    }

    function drawTimeline(payload) {
      const canvas = $("timelineCanvas");
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      const n = payload.segments.length;
      if (!n) return;
      const blockW = (w - 16) / n;
      payload.segments.forEach((seg, i) => {
        const x = 8 + i * blockW;
        ctx.fillStyle = CLASS_COLORS[seg.class_id] || "#1f4e79";
        ctx.globalAlpha = i === selectedSegment ? 1.0 : 0.86;
        ctx.fillRect(x, 22, blockW - 4, h - 70);
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = "#ffffff";
        ctx.font = "600 12px Arial";
        ctx.textAlign = "center";
        ctx.fillText(`${seg.class_name[0]}${seg.confidence}%`, x + (blockW - 4) / 2, h / 2);
        if (payload.has_ground_truth) {
          ctx.fillStyle = seg.class_id === seg.ground_truth_class_id ? "#14866d" : "#b32424";
          ctx.beginPath();
          ctx.arc(x + (blockW - 4) / 2, h - 22, 6, 0, 2 * Math.PI);
          ctx.fill();
        }
        ctx.fillStyle = "#54595d";
        ctx.font = "11px Arial";
        ctx.textAlign = "center";
        ctx.fillText(`${seg.start_seconds.toFixed(1)}s`, x + (blockW - 4) / 2, 14);
      });
      ctx.strokeStyle = "#1f4e79";
      ctx.lineWidth = 2;
      const x = 8 + selectedSegment * blockW;
      ctx.strokeRect(x - 1, 21, blockW - 2, h - 68);

      canvas.onclick = (ev) => {
        const rect = canvas.getBoundingClientRect();
        const px = (ev.clientX - rect.left) * canvas.width / rect.width;
        const idx = Math.floor((px - 8) / blockW);
        if (idx >= 0 && idx < n) selectSegment(idx);
      };
    }

    function drawProbabilityChart(payload) {
      const canvas = $("probChart");
      const ctx = canvas.getContext("2d");
      const w = canvas.width, h = canvas.height;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      const m = { l: 56, r: 28, t: 18, b: 38 };
      const innerW = w - m.l - m.r;
      const innerH = h - m.t - m.b;
      ctx.strokeStyle = "#d0d6dd";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i <= 4; i++) {
        const y = m.t + (i / 4) * innerH;
        ctx.moveTo(m.l, y);
        ctx.lineTo(m.l + innerW, y);
      }
      ctx.stroke();
      ctx.fillStyle = "#4f5b69";
      ctx.font = "11px Arial";
      ctx.textAlign = "right";
      ["100", "75", "50", "25", "0"].forEach((label, i) => {
        ctx.fillText(label + "%", m.l - 6, m.t + (i / 4) * innerH + 4);
      });
      ctx.textAlign = "center";
      const n = payload.segments.length;
      payload.segments.forEach((seg, i) => {
        const x = m.l + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
        ctx.fillText(`${seg.start_seconds.toFixed(1)}s`, x, m.t + innerH + 18);
      });

      for (let c = 0; c < 3; c++) {
        ctx.strokeStyle = CLASS_COLORS[c];
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        payload.segments.forEach((seg, i) => {
          const probs = seg.raw_probabilities || [0, 0, 0];
          const x = m.l + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
          const y = m.t + innerH - (probs[c] / 100) * innerH;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();

        payload.segments.forEach((seg, i) => {
          const probs = seg.raw_probabilities || [0, 0, 0];
          const x = m.l + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
          const y = m.t + innerH - (probs[c] / 100) * innerH;
          ctx.fillStyle = CLASS_COLORS[c];
          ctx.beginPath();
          ctx.arc(x, y, seg.gate_applied && c === 2 ? 6 : 3.5, 0, 2 * Math.PI);
          ctx.fill();
        });
      }
      // Legend
      ctx.font = "12px Arial";
      ctx.textAlign = "left";
      const legendY = m.t + 4;
      let legendX = m.l + 4;
      CLASS_NAMES.forEach((name, c) => {
        ctx.fillStyle = CLASS_COLORS[c];
        ctx.fillRect(legendX, legendY, 12, 12);
        ctx.fillStyle = "#1f2933";
        ctx.fillText(`  ${name}`, legendX + 8, legendY + 11);
        legendX += 110;
      });
    }

    function drawClassBars(payload) {
      const canvas = $("classBarChart");
      const ctx = canvas.getContext("2d");
      const w = canvas.width, h = canvas.height;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      const counts = payload.aggregate.counts;
      const max = Math.max(1, ...CLASS_KEYS.map(k => counts[k] || 0));
      const barW = (w - 80) / CLASS_KEYS.length;
      CLASS_KEYS.forEach((k, i) => {
        const count = counts[k] || 0;
        const barH = (count / max) * (h - 60);
        const x = 40 + i * barW + 14;
        ctx.fillStyle = CLASS_COLORS[i];
        ctx.fillRect(x, h - 30 - barH, barW - 28, barH);
        ctx.fillStyle = "#1f2933";
        ctx.font = "700 13px Arial";
        ctx.textAlign = "center";
        ctx.fillText(`${count}`, x + (barW - 28) / 2, h - 36 - barH);
        ctx.font = "12px Arial";
        ctx.fillText(CLASS_NAMES[i], x + (barW - 28) / 2, h - 10);
      });
    }

    function drawRmsChart(payload) {
      const canvas = $("rmsChart");
      const ctx = canvas.getContext("2d");
      const w = canvas.width, h = canvas.height;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      const m = { l: 50, r: 18, t: 14, b: 32 };
      const innerW = w - m.l - m.r;
      const innerH = h - m.t - m.b;
      ctx.strokeStyle = "#d0d6dd";
      ctx.beginPath();
      for (let i = 0; i <= 4; i++) {
        const y = m.t + (i / 4) * innerH;
        ctx.moveTo(m.l, y);
        ctx.lineTo(m.l + innerW, y);
      }
      ctx.stroke();
      const maxRms = Math.max(0.05, ...payload.segments.map(s => s.audio_stats.rms));
      ctx.fillStyle = "#4f5b69";
      ctx.font = "11px Arial";
      ctx.textAlign = "right";
      for (let i = 0; i <= 4; i++) {
        const val = (maxRms * (1 - i / 4)).toFixed(2);
        ctx.fillText(val, m.l - 5, m.t + (i / 4) * innerH + 4);
      }
      ctx.strokeStyle = "#3573a6";
      ctx.lineWidth = 2;
      ctx.beginPath();
      const n = payload.segments.length;
      payload.segments.forEach((seg, i) => {
        const x = m.l + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
        const y = m.t + innerH - (seg.audio_stats.rms / maxRms) * innerH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = "#3573a6";
      payload.segments.forEach((seg, i) => {
        const x = m.l + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
        const y = m.t + innerH - (seg.audio_stats.rms / maxRms) * innerH;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    function drawSegmentWaveform(values) {
      const canvas = $("segmentWaveform");
      const ctx = canvas.getContext("2d");
      const w = canvas.width, h = canvas.height;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "#d0d6dd";
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();
      if (!values || values.length === 0) return;
      let maxAbs = 1e-6;
      for (const v of values) maxAbs = Math.max(maxAbs, Math.abs(v));
      ctx.strokeStyle = "#1f4e79";
      ctx.lineWidth = 1.7;
      ctx.beginPath();
      values.forEach((v, i) => {
        const x = values.length === 1 ? 0 : i * (w - 1) / (values.length - 1);
        const y = h / 2 - (v / maxAbs) * (h * 0.44);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    function colorMap(t) {
      t = Math.max(0, Math.min(1, t));
      const r = Math.round(35 + 220 * t);
      const g = Math.round(40 + 130 * Math.sin(t * Math.PI));
      const b = Math.round(80 + 160 * (1 - t));
      return `rgb(${r},${g},${b})`;
    }

    function drawSpectrogram(values) {
      const canvas = $("specCanvas");
      const ctx = canvas.getContext("2d");
      const w = canvas.width, h = canvas.height;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      if (!values || values.length !== 4096) return;
      let lo = Infinity, hi = -Infinity;
      for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
      const span = Math.max(hi - lo, 1e-6);
      const cellW = w / 64, cellH = h / 64;
      for (let row = 0; row < 64; row++) {
        for (let col = 0; col < 64; col++) {
          const srcRow = 63 - row;
          const t = (values[srcRow * 64 + col] - lo) / span;
          ctx.fillStyle = colorMap(t);
          ctx.fillRect(col * cellW, row * cellH, Math.ceil(cellW), Math.ceil(cellH));
        }
      }
    }

    function renderResult(payload) {
      lastPayload = payload;
      selectedSegment = Math.min(selectedSegment, payload.segments.length - 1);
      if (selectedSegment < 0) selectedSegment = 0;
      renderAggregate(payload);
      renderSegments(payload);
      drawTimeline(payload);
      drawProbabilityChart(payload);
      drawClassBars(payload);
      drawRmsChart(payload);
      selectSegment(selectedSegment);
      if (payload.audio_data_url) {
        $("audioPlayer").src = payload.audio_data_url;
      }
    }

    async function loadPorts() {
      const response = await fetch("/api/ports");
      const data = await response.json();
      for (const sel of [$("serialPort"), $("loopSerialPort")]) {
        sel.innerHTML = "";
        const ports = data.ports.length ? data.ports : [data.default_port];
        for (const p of ports) {
          const opt = document.createElement("option");
          opt.value = p;
          opt.textContent = p;
          if (p === data.default_port) opt.selected = true;
          sel.appendChild(opt);
        }
      }
      const port = $("serialPort").value || "COM6";
      const connected = data.ports.includes(port);
      $("uartStatus").className = `uart-pill ${connected ? "connected" : "disconnected"}`;
      $("uartStatus").textContent = `${port} ${connected ? "Connected" : "Disconnected"}`;
    }

    async function loadGenericSources() {
      const response = await fetch("/api/generic");
      const data = await response.json();
      $("genericSource").innerHTML = "";
      data.samples.forEach((s, idx) => {
        const opt = document.createElement("option");
        opt.value = String(idx);
        opt.textContent = s.name;
        $("genericSource").appendChild(opt);
      });
      if (!data.samples.length) {
        const opt = document.createElement("option");
        opt.value = "0";
        opt.textContent = "No generic samples found";
        $("genericSource").appendChild(opt);
      }
    }

    async function classifyFormData(body, url = "/api/classify") {
      const t0 = performance.now();
      const response = await fetch(url, { method: "POST", body });
      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Classification failed");
      }
      const elapsed = performance.now() - t0;
      log([
        `Source: ${data.file_name}`,
        `Serial: ${data.port} @ ${data.baud}`,
        `Segments: ${data.n_segments} x ${data.window_seconds} s = ${data.audio_seconds.toFixed(1)} s`,
        `Round trip: ${Math.round(elapsed)} ms (server worker)`,
        data.aggregate.accuracy_pct !== null && data.aggregate.accuracy_pct !== undefined
          ? `Ground-truth accuracy: ${data.aggregate.accuracy_pct.toFixed(1)}%`
          : `No ground truth provided.`,
      ].join("\n"));
      renderResult(data);
      return data;
    }

    $("classifyForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      $("submitBtn").disabled = true;
      setProgress(0, 1);
      log("Uploading audio to STM32 for on-chip DSP + inference...");
      try {
        const form = new FormData($("classifyForm"));
        await classifyFormData(form);
        setProgress(1, 1);
        setTimeout(() => setProgress(0, 0), 500);
      } catch (e) {
        log(e.message, true);
        setProgress(0, 0);
      } finally {
        $("submitBtn").disabled = false;
      }
    });

    $("clearBtn").addEventListener("click", () => {
      $("classifyForm").reset();
      log("Ready.");
    });

    $("customTab").addEventListener("click", () => setMode("custom"));
    $("loopTab").addEventListener("click", () => setMode("loop"));

    $("startLoopBtn").addEventListener("click", () => {
      loopNext = Number($("genericSource").value || 0);
      $("startLoopBtn").disabled = true;
      $("stopLoopBtn").disabled = false;
      log("Generic loop running.");
      runLoopOnce();
      const intervalMs = Math.max(1, Number($("loopInterval").value || 3)) * 1000;
      loopTimer = setInterval(runLoopOnce, intervalMs);
    });

    $("stopLoopBtn").addEventListener("click", () => {
      if (loopTimer) clearInterval(loopTimer);
      loopTimer = null;
      $("startLoopBtn").disabled = false;
      $("stopLoopBtn").disabled = true;
      log("Generic loop stopped.");
    });

    async function runLoopOnce() {
      if (loopBusy) return;
      loopBusy = true;
      const body = new FormData();
      body.set("port", $("loopSerialPort").value);
      body.set("baud", $("loopBaud").value);
      body.set("index", String(loopNext));
      try {
        const data = await classifyFormData(body, "/api/classify_generic");
        loopNext = data.next_index;
      } catch (e) {
        log(e.message, true);
      } finally {
        loopBusy = false;
      }
    }

    $("exportBtn").addEventListener("click", () => {
      if (!lastPayload) return;
      const header = ["seg", "start_s", "class_id", "class_name", "ground_truth",
                      "confidence", "p_absent", "p_present", "p_unknown",
                      "dsp_ms", "infer_ms", "uart_upload_ms", "uart_return_ms",
                      "rms", "peak", "zcr", "gate_applied"];
      const rows = lastPayload.segments.map((s, i) => [
        i + 1, s.start_seconds.toFixed(2), s.class_id, s.class_name,
        s.ground_truth_class_name || "",
        s.confidence,
        (s.raw_probabilities || [0,0,0])[0],
        (s.raw_probabilities || [0,0,0])[1],
        (s.raw_probabilities || [0,0,0])[2],
        s.latency.stm32_dsp_ms, s.latency.stm32_inference_ms,
        s.latency.uart_upload_ms, s.latency.uart_return_ms,
        s.audio_stats.rms, s.audio_stats.peak, s.audio_stats.zcr,
        s.gate_applied ? 1 : 0,
      ]);
      const csv = [header, ...rows].map(r => r.map(v => `"${String(v).replaceAll('"','""')}"`).join(",")).join("\n");
      const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = `digital-stethoscope-${(lastPayload.file_name || 'session').replace(/\W+/g,'_')}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    });

    loadPorts().catch(() => {
      $("serialPort").innerHTML = '<option value="COM6">COM6</option>';
      $("loopSerialPort").innerHTML = '<option value="COM6">COM6</option>';
      $("uartStatus").className = "uart-pill disconnected";
      $("uartStatus").textContent = "COM Disconnected";
    });
    setInterval(() => loadPorts().catch(() => {}), 5000);
    loadGenericSources().catch(() => log("Could not load generic samples.", true));
  </script>
</body>
</html>
"""


def fit_audio_window(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) < WIN_SAMPLES:
        return np.pad(audio, (0, WIN_SAMPLES - len(audio))).astype(np.float32)
    return audio[:WIN_SAMPLES].astype(np.float32)


def segment_audio(audio: np.ndarray) -> list[np.ndarray]:
    """Split a long mono float32 waveform into successive 2-second windows."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    n_windows = max(1, int(np.ceil(len(audio) / WIN_SAMPLES)))
    windows = []
    for i in range(n_windows):
        start = i * WIN_SAMPLES
        end = start + WIN_SAMPLES
        chunk = audio[start:end]
        if len(chunk) < WIN_SAMPLES:
            chunk = np.pad(chunk, (0, WIN_SAMPLES - len(chunk)))
        windows.append(chunk.astype(np.float32))
    return windows


def load_wav(path: Path) -> tuple[np.ndarray, float]:
    audio, _ = librosa.load(str(path), sr=SR_TARGET, mono=True)
    audio = audio.astype(np.float32)
    return audio, float(len(audio) / SR_TARGET)


def load_npy(path: Path) -> tuple[np.ndarray, float]:
    arr = np.load(path, allow_pickle=False).astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(".npy input must be a 1D audio waveform")
    return arr, float(len(arr) / SR_TARGET)


def preprocess_input_file(path: Path) -> tuple[np.ndarray, float]:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return load_wav(path)
    if suffix == ".npy":
        return load_npy(path)
    raise ValueError("Unsupported input. Use .wav or .npy")


def waveform_preview(audio: np.ndarray | None, max_points: int = 900) -> list[float]:
    if audio is None or len(audio) == 0:
        return []
    if len(audio) <= max_points:
        return [float(x) for x in audio]
    idx = np.linspace(0, len(audio) - 1, max_points).astype(np.int32)
    return [float(x) for x in audio[idx]]


def audio_data_url(audio: np.ndarray | None) -> str | None:
    if audio is None or len(audio) == 0:
        return None
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SR_TARGET)
        wav.writeframes(pcm.tobytes())
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def generic_samples(limit: int = 80) -> list[Path]:
    found: list[Path] = []
    for d in GENERIC_AUDIO_DIRS:
        if d.exists():
            found.extend(sorted(d.rglob("*.wav"))[:limit])
            if len(found) >= limit:
                break
    return found[:limit]


def recv_magic(ser: serial.Serial, magic: int, timeout: float = 8.0) -> bytes:
    deadline = time.time() + timeout
    while time.time() < deadline:
        byte = ser.read(1)
        if byte and byte[0] == magic:
            return byte
    raise TimeoutError(f"Timed out waiting for STM32 byte 0x{magic:02X}")


def recv_exact(ser: serial.Serial, n_bytes: int, timeout: float = 15.0) -> bytes:
    deadline = time.time() + timeout
    chunks = bytearray()
    while len(chunks) < n_bytes and time.time() < deadline:
        chunk = ser.read(n_bytes - len(chunks))
        if chunk:
            chunks.extend(chunk)
    if len(chunks) != n_bytes:
        raise TimeoutError(f"Timed out reading {n_bytes} bytes from STM32")
    return bytes(chunks)


def recv_extended_response(ser: serial.Serial) -> tuple[dict, np.ndarray, float]:
    recv_magic(ser, RESPONSE_MAGIC, timeout=8.0)
    rest = recv_exact(ser, RESPONSE_HEADER_BYTES - 1, timeout=4.0)
    response_start = time.perf_counter()
    raw_spec = recv_exact(ser, RESPONSE_SPEC_BYTES, timeout=15.0)
    return_ms = (time.perf_counter() - response_start) * 1000.0

    class_id, conf, p_a, p_p, p_u, gate = rest[:6]
    dsp_ms, infer_ms, rms_q15, peak_q15, zcr = struct.unpack("<HHHHH", rest[6:16])
    spec = np.frombuffer(raw_spec, dtype="<f4").reshape(SPEC_SIZE, SPEC_SIZE).copy()

    fields = {
        "class_id": int(class_id),
        "confidence": int(conf),
        "raw_probabilities": [int(p_a), int(p_p), int(p_u)],
        "gate_applied": int(gate),
        "stm32_dsp_ms": int(dsp_ms),
        "stm32_inference_ms": int(infer_ms),
        "rms": float(rms_q15) / 32768.0,
        "peak": float(peak_q15) / 32768.0,
        "zcr": int(zcr),
    }
    return fields, spec, return_ms


def classify_window(ser: serial.Serial, audio: np.ndarray, baud: int) -> dict:
    pcm = (np.clip(fit_audio_window(audio), -1.0, 1.0) * 32767.0).astype("<i2")
    ser.reset_input_buffer()
    ser.write(b"A")
    ser.flush()
    upload_start = time.perf_counter()
    # STM32 emits an ACK after every 128 received samples PLUS one final ACK
    # after the loop. With 8000 samples and 128-sample host chunks Python's
    # range yields exactly ceil(8000/128) = 63 iterations, which exactly
    # matches 62 inline ACKs + 1 final ACK = 63 total. One ACK per iteration.
    for offset in range(0, pcm.size, HOST_SUBCHUNK_ELEMENTS):
        chunk = pcm[offset:offset + HOST_SUBCHUNK_ELEMENTS]
        ser.write(chunk.tobytes(order="C"))
        ser.flush()
        recv_magic(ser, UPLOAD_ACK_MAGIC, timeout=4.0)
    upload_ms = (time.perf_counter() - upload_start) * 1000.0

    fields, spec, return_ms = recv_extended_response(ser)
    theoretical_upload_ms = (pcm.nbytes * 10.0 / float(baud)) * 1000.0
    theoretical_return_ms = (RESPONSE_SPEC_BYTES * 10.0 / float(baud)) * 1000.0

    return {
        **fields,
        "spec": spec,
        "uart_upload_ms": round(upload_ms, 1),
        "uart_return_ms": round(return_ms, 1),
        "uart_upload_theoretical_ms": round(theoretical_upload_ms, 1),
        "uart_return_theoretical_ms": round(theoretical_return_ms, 1),
    }


def classify_all_windows(audio: np.ndarray, port: str, baud: int,
                         lock: threading.Lock) -> list[dict]:
    windows = segment_audio(audio)
    results: list[dict] = []
    with lock:
        ser = serial.Serial()
        ser.port = port
        ser.baudrate = baud
        ser.timeout = 0.2
        ser.write_timeout = 5.0
        ser.dtr = False
        ser.rts = False
        with ser:
            time.sleep(0.6)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            for window in windows:
                results.append(classify_window(ser, window, baud))
    return results


def parse_labels_json(raw_bytes: bytes) -> dict | None:
    if not raw_bytes:
        return None
    try:
        text = raw_bytes.decode("utf-8")
        return json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def map_labels(payload_segments: list[dict], labels: dict | None) -> bool:
    if not labels:
        for seg in payload_segments:
            seg["ground_truth_class_id"] = None
            seg["ground_truth_class_name"] = None
        return False

    label_segments = labels.get("segments") or []
    has = False
    for seg in payload_segments:
        idx = seg["index"]
        if idx < len(label_segments):
            entry = label_segments[idx]
            class_name = entry.get("class_name")
            class_id = entry.get("class_id")
            if class_name in CLASS_NAMES:
                cid = CLASS_NAMES.index(class_name)
            elif isinstance(class_id, int) and 0 <= class_id < len(CLASS_NAMES):
                cid = class_id
            else:
                cid = None
            seg["ground_truth_class_id"] = cid
            seg["ground_truth_class_name"] = CLASS_NAMES[cid] if cid is not None else None
            if cid is not None:
                has = True
        else:
            seg["ground_truth_class_id"] = None
            seg["ground_truth_class_name"] = None
    return has


def assemble_payload(*, source_name: str, port: str, baud: int,
                     audio: np.ndarray, duration: float, results: list[dict],
                     labels: dict | None) -> dict:
    segments: list[dict] = []
    for i, res in enumerate(results):
        start = i * WIN_SEC
        end = start + WIN_SEC
        window_audio = audio[i * WIN_SAMPLES:(i + 1) * WIN_SAMPLES]
        if len(window_audio) == 0:
            window_audio = np.zeros(WIN_SAMPLES, dtype=np.float32)
        segments.append({
            "index": i,
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "class_id": res["class_id"],
            "class_name": CLASS_NAMES[res["class_id"]] if 0 <= res["class_id"] < len(CLASS_NAMES) else "Invalid",
            "confidence": res["confidence"],
            "raw_probabilities": res["raw_probabilities"],
            "gate_applied": bool(res["gate_applied"]),
            "audio_stats": {
                "rms": round(res["rms"], 4),
                "peak": round(res["peak"], 4),
                "zcr": int(res["zcr"]),
            },
            "latency": {
                "stm32_dsp_ms": res["stm32_dsp_ms"],
                "stm32_inference_ms": res["stm32_inference_ms"],
                "uart_upload_ms": res["uart_upload_ms"],
                "uart_return_ms": res["uart_return_ms"],
                "uart_upload_theoretical_ms": res["uart_upload_theoretical_ms"],
                "uart_return_theoretical_ms": res["uart_return_theoretical_ms"],
            },
            "spectrogram": [float(x) for x in res["spec"].reshape(-1)],
            "waveform_preview": waveform_preview(window_audio),
            "ground_truth_class_id": None,
            "ground_truth_class_name": None,
        })

    has_truth = map_labels(segments, labels)

    # Aggregate stats
    if segments:
        class_counter = Counter(CLASS_KEYS[seg["class_id"]] for seg in segments
                                if 0 <= seg["class_id"] < len(CLASS_KEYS))
        counts = {k: class_counter.get(k, 0) for k in CLASS_KEYS}
        dominant_key = max(counts, key=counts.get)
        dominant_class_id = CLASS_KEYS.index(dominant_key)
        dominant_segments = [s for s in segments if s["class_id"] == dominant_class_id]
        dominant_confidence = (sum(s["confidence"] for s in dominant_segments) /
                               max(1, len(dominant_segments)))

        avg_dsp = sum(s["latency"]["stm32_dsp_ms"] for s in segments) / len(segments)
        avg_inf = sum(s["latency"]["stm32_inference_ms"] for s in segments) / len(segments)
        avg_up = sum(s["latency"]["uart_upload_ms"] for s in segments) / len(segments)
        avg_ret = sum(s["latency"]["uart_return_ms"] for s in segments) / len(segments)

        accuracy_pct = None
        if has_truth:
            usable = [s for s in segments if s["ground_truth_class_id"] is not None]
            if usable:
                correct = sum(1 for s in usable if s["class_id"] == s["ground_truth_class_id"])
                accuracy_pct = correct * 100.0 / len(usable)
    else:
        counts = {k: 0 for k in CLASS_KEYS}
        dominant_class_id = 0
        dominant_confidence = 0
        avg_dsp = avg_inf = avg_up = avg_ret = 0
        accuracy_pct = None

    return {
        "ok": True,
        "file_name": source_name,
        "port": port,
        "baud": baud,
        "sample_rate": SR_TARGET,
        "window_seconds": WIN_SEC,
        "n_segments": len(segments),
        "audio_seconds": duration,
        "classes": CLASS_NAMES,
        "segments": segments,
        "has_ground_truth": has_truth,
        "audio_data_url": audio_data_url(audio[:WIN_SAMPLES * len(segments)]) if segments else None,
        "aggregate": {
            "counts": counts,
            "dominant_class_id": dominant_class_id,
            "dominant_confidence": round(dominant_confidence, 1),
            "avg_dsp_ms": round(avg_dsp, 1),
            "avg_inference_ms": round(avg_inf, 1),
            "avg_uart_upload_ms": round(avg_up, 1),
            "avg_uart_return_ms": round(avg_ret, 1),
            "accuracy_pct": accuracy_pct,
        },
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "DigitalStethoscopeDashboard/2.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self.send_html(HTML)
        if parsed.path == "/api/ports":
            ports = [p.device for p in list_ports.comports()]
            return self.send_json({"ports": ports, "default_port": self.server.default_serial_port})
        if parsed.path == "/api/generic":
            samples = [
                {"name": p.name, "path": str(p.relative_to(ROOT))}
                for p in generic_samples()
            ]
            return self.send_json({"samples": samples})
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/classify":
            return self.handle_custom_classify()
        if parsed.path == "/api/classify_generic":
            return self.handle_generic_classify()
        self.send_error(404)

    def parse_form(self) -> cgi.FieldStorage:
        return cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )

    def handle_custom_classify(self) -> None:
        try:
            form = self.parse_form()
            file_item = form["file"] if "file" in form else None
            if file_item is None or not file_item.filename:
                raise ValueError("Choose a WAV or NPY file first")

            port = form.getfirst("port", self.server.default_serial_port).strip()
            baud = int(form.getfirst("baud", str(self.server.default_baud)))
            if not port:
                raise ValueError("Serial port is required")

            suffix = Path(file_item.filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_item.file.read())
                tmp_path = Path(tmp.name)
            try:
                audio, duration = preprocess_input_file(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)

            labels = None
            labels_item = form["labels"] if "labels" in form else None
            if labels_item is not None and labels_item.filename:
                labels = parse_labels_json(labels_item.file.read())

            # Auto-look-up of labels JSON for known demo/presentation files
            if labels is None:
                stem = Path(file_item.filename).stem
                for candidate_dir in (PRESENTATION_DIR, DEMO_DIR):
                    sibling_labels = candidate_dir / f"{stem}_labels.json"
                    if sibling_labels.exists():
                        labels = parse_labels_json(sibling_labels.read_bytes())
                        break

            results = classify_all_windows(audio, port, baud, self.server.serial_lock)
            payload = assemble_payload(
                source_name=Path(file_item.filename).name,
                port=port, baud=baud, audio=audio,
                duration=duration, results=results, labels=labels,
            )
            return self.send_json(payload)
        except Exception as exc:
            return self.send_json({"ok": False, "error": str(exc)}, status=400)

    def handle_generic_classify(self) -> None:
        try:
            form = self.parse_form()
            samples = generic_samples()
            if not samples:
                raise ValueError("No generic WAV samples found")

            index = int(form.getfirst("index", "0")) % len(samples)
            port = form.getfirst("port", self.server.default_serial_port).strip()
            baud = int(form.getfirst("baud", str(self.server.default_baud)))
            path = samples[index]

            audio, duration = preprocess_input_file(path)
            sibling_labels = path.with_name(path.stem + "_labels.json")
            labels = parse_labels_json(sibling_labels.read_bytes()) if sibling_labels.exists() else None

            results = classify_all_windows(audio, port, baud, self.server.serial_lock)
            payload = assemble_payload(
                source_name=path.name,
                port=port, baud=baud, audio=audio,
                duration=duration, results=results, labels=labels,
            )
            payload["next_index"] = (index + 1) % len(samples)
            return self.send_json(payload)
        except Exception as exc:
            return self.send_json({"ok": False, "error": str(exc)}, status=400)

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))

    def send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_json(self, data: dict, status: int = 200) -> None:
        encoded = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Digital Stethoscope dashboard (v2 CirCor)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--port", default="COM6", help="Default STM32 serial port")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.http_port), DashboardHandler)
    server.default_serial_port = args.port
    server.default_baud = args.baud
    server.serial_lock = threading.Lock()

    print(f"Dashboard: http://{args.host}:{args.http_port}")
    print(f"Default STM32 serial: {args.port} @ {args.baud}")
    server.serve_forever()


if __name__ == "__main__":
    main()
