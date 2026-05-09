#!/usr/bin/env python3
"""
Local STM32 inference dashboard.

Serves a dependency-light web UI and bridges browser controls to the STM32
console UART. The STM32 runs ML locally; this script only starts/stops synthetic
injection and visualizes UART logs.
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import threading
import time
from collections import Counter, deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

try:
    import serial
    from serial.tools import list_ports
except Exception as exc:  # pragma: no cover - displayed at runtime
    serial = None
    list_ports = None
    SERIAL_IMPORT_ERROR = exc
else:
    SERIAL_IMPORT_ERROR = None


CLASS_RE = re.compile(
    r"Class:\s+(?P<class>[A-Za-z0-9]+)\s+Confidence:\s+(?P<conf>\d+)%\s+Latency:\s+(?P<lat>\d+)ms"
)
INPUT_RE = re.compile(r"Synthetic input string:\s+(?P<input>[A-Za-z0-9_]+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

CLASS_NAMES = ["Normal", "SysMurmur", "DiaMurmur", "S3Gallop"]


def clean_uart_line(raw: bytes) -> str:
    text = raw.decode(errors="ignore").strip()
    text = ANSI_RE.sub("", text)

    return "".join(ch for ch in text if ch == "\t" or ch == " " or 0x20 <= ord(ch) <= 0x7E)


class DashboardState:
    def __init__(self, port: str, baud: int):
        self.port = port
        self.baud = baud
        self.serial = None
        self.serial_lock = threading.Lock()
        self.clients: list[queue.Queue[str]] = []
        self.clients_lock = threading.Lock()
        self.running = False
        self.connected = False
        self.status = "disconnected"
        self.last_input = "-"
        self.latest = None
        self.history = deque(maxlen=120)
        self.counts = Counter()
        self.logs = deque(maxlen=200)
        self.stop_event = threading.Event()
        self.reader_thread: threading.Thread | None = None

    def start_reader(self) -> None:
        if self.reader_thread and self.reader_thread.is_alive():
            return
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

    def _connect(self):
        if serial is None:
            raise RuntimeError(f"pyserial unavailable: {SERIAL_IMPORT_ERROR}")
        ser = serial.Serial(self.port, self.baud, timeout=0.2)
        ser.reset_input_buffer()
        return ser

    def _reader_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                if self.serial is None or not self.serial.is_open:
                    self.status = f"connecting to {self.port}"
                    self.broadcast_state()
                    with self.serial_lock:
                        self.serial = self._connect()
                    self.connected = True
                    self.status = f"connected to {self.port}"
                    self.add_log(self.status)
                    self.broadcast_state()

                raw = self.serial.readline()
                if raw:
                    line = clean_uart_line(raw)
                    if line:
                        self.handle_line(line)
            except Exception as exc:
                self.connected = False
                self.status = f"serial error: {exc}"
                self.add_log(self.status)
                self.broadcast_state()
                with self.serial_lock:
                    try:
                        if self.serial is not None:
                            self.serial.close()
                    except Exception:
                        pass
                    self.serial = None
                time.sleep(1.0)

    def add_log(self, line: str) -> None:
        self.logs.append({"ts": time.time(), "line": line})
        self.broadcast({"type": "log", "line": line, "ts": time.time()})

    def handle_line(self, line: str) -> None:
        self.add_log(line)

        input_match = INPUT_RE.search(line)
        if input_match:
            self.last_input = input_match.group("input")
            self.broadcast_state()

        class_match = CLASS_RE.search(line)
        if class_match:
            self.running = True
            item = {
                "ts": time.time(),
                "input": self.last_input,
                "class": class_match.group("class"),
                "confidence": int(class_match.group("conf")),
                "latency": int(class_match.group("lat")),
            }
            self.latest = item
            self.history.append(item)
            self.counts[item["class"]] += 1
            self.broadcast({"type": "inference", "item": item, "state": self.snapshot()})

    def command(self, cmd: bytes) -> bool:
        self.start_reader()
        deadline = time.time() + 3.0
        while time.time() < deadline:
            with self.serial_lock:
                ser = self.serial
                if ser is not None and ser.is_open:
                    ser.write(cmd)
                    ser.flush()
                    self.running = cmd == b"S"
                    self.broadcast_state()
                    return True
            time.sleep(0.05)
        return False

    def snapshot(self) -> dict:
        return {
            "port": self.port,
            "baud": self.baud,
            "connected": self.connected,
            "running": self.running,
            "status": self.status,
            "latest": self.latest,
            "history": list(self.history),
            "counts": dict(self.counts),
            "logs": list(self.logs)[-80:],
            "classes": CLASS_NAMES,
        }

    def register(self) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue(maxsize=100)
        with self.clients_lock:
            self.clients.append(q)
        q.put(f"data: {json.dumps({'type': 'state', 'state': self.snapshot()})}\n\n")
        return q

    def unregister(self, q: queue.Queue[str]) -> None:
        with self.clients_lock:
            if q in self.clients:
                self.clients.remove(q)

    def broadcast_state(self) -> None:
        self.broadcast({"type": "state", "state": self.snapshot()})

    def broadcast(self, payload: dict) -> None:
        msg = f"data: {json.dumps(payload)}\n\n"
        with self.clients_lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>STM32 ML Inference Dashboard</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #667085;
      --line: #d7dde5;
      --accent: #1677ff;
      --ok: #14804a;
      --warn: #b54708;
      --bad: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 { margin: 0; font-size: 20px; font-weight: 700; letter-spacing: 0; }
    .sub { color: var(--muted); font-size: 13px; margin-top: 3px; }
    .actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    button {
      height: 38px;
      padding: 0 14px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--ink);
      font-weight: 650;
      cursor: pointer;
    }
    button.primary { background: var(--accent); border-color: var(--accent); color: white; }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    main {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 16px;
      padding: 16px;
      max-width: 1320px;
      margin: 0 auto;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }
    .span { grid-column: 1 / -1; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 86px;
    }
    .label { color: var(--muted); font-size: 12px; font-weight: 650; text-transform: uppercase; }
    .value { margin-top: 8px; font-size: 24px; font-weight: 760; overflow-wrap: anywhere; }
    .status { display: inline-flex; align-items: center; gap: 8px; color: var(--muted); font-size: 13px; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--bad); }
    .dot.on { background: var(--ok); }
    .bars { display: grid; gap: 11px; margin-top: 10px; }
    .bar-row { display: grid; grid-template-columns: 112px 1fr 42px; align-items: center; gap: 10px; font-size: 13px; }
    .track { height: 16px; background: #edf1f5; border-radius: 4px; overflow: hidden; }
    .fill { height: 100%; width: 0%; background: var(--accent); transition: width 180ms ease; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 9px 8px; border-bottom: 1px solid var(--line); text-align: left; }
    th { color: var(--muted); font-size: 12px; font-weight: 700; }
    .log {
      height: 290px;
      overflow: auto;
      background: #111827;
      color: #e5e7eb;
      border-radius: 6px;
      padding: 10px;
      font-family: Consolas, ui-monospace, monospace;
      font-size: 12px;
      line-height: 1.45;
    }
    .signal-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      align-items: stretch;
    }
    canvas {
      display: block;
      width: 100%;
      height: 240px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfe;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }
    .small { color: var(--muted); font-size: 12px; margin-top: 8px; }
    @media (max-width: 920px) {
      header { align-items: flex-start; flex-direction: column; }
      main { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .signal-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>STM32 ML Inference Dashboard</h1>
      <div class="sub">Synthetic PCG injection -> STM32 DSP -> local TFLite Micro classification</div>
    </div>
    <div class="actions">
      <span class="status"><span id="dot" class="dot"></span><span id="status">connecting</span></span>
      <button id="start" class="primary">Start Injection</button>
      <button id="pause">Pause</button>
    </div>
  </header>
  <main>
    <section class="span metrics">
      <div class="metric"><div class="label">Latest Class</div><div id="latestClass" class="value">-</div></div>
      <div class="metric"><div class="label">Confidence</div><div id="confidence" class="value">-</div></div>
      <div class="metric"><div class="label">Latency</div><div id="latency" class="value">-</div></div>
      <div class="metric"><div class="label">Injected Input</div><div id="input" class="value">-</div></div>
    </section>
    <section class="span signal-grid">
      <div>
        <div class="panel-head">
          <div class="label">DSP Input Waveform</div>
          <button id="playAudio">Play Heartbeat</button>
        </div>
        <canvas id="waveform" width="900" height="260"></canvas>
        <div class="small">Synthetic 2-second PCM window rendered at 4 kHz before the STM32 DSP stage.</div>
      </div>
      <div>
        <div class="panel-head">
          <div class="label">ML Input Transform</div>
        </div>
        <canvas id="spectrogram" width="900" height="260"></canvas>
        <div class="small">Mel-style time-frequency view corresponding to the spectrogram tensor sent into TFLite Micro.</div>
      </div>
    </section>
    <section>
      <div class="label">Classification Counts</div>
      <div id="bars" class="bars"></div>
    </section>
    <section>
      <div class="label">Recent Inferences</div>
      <table>
        <thead><tr><th>Time</th><th>Input</th><th>Class</th><th>Conf</th><th>Latency</th></tr></thead>
        <tbody id="history"></tbody>
      </table>
    </section>
    <section class="span">
      <div class="label" style="margin-bottom:10px;">UART Log</div>
      <div id="log" class="log"></div>
    </section>
  </main>
  <script>
    const state = { classes: ["Normal", "SysMurmur", "DiaMurmur", "S3Gallop"], counts: {}, history: [] };
    const $ = (id) => document.getElementById(id);
    const SAMPLE_RATE = 4000;
    const WINDOW_SECONDS = 2;
    let renderedInput = null;
    let latestSamples = new Float32Array(SAMPLE_RATE * WINDOW_SECONDS);

    const synthScripts = {
      NORMAL: "0.05,0.10,100,0.40;0.45,0.08,120,0.24;1.05,0.10,100,0.40;1.45,0.08,120,0.24",
      SYSTOLIC: "0.05,0.10,100,0.32;0.14,0.30,260,0.35;0.45,0.08,120,0.20;1.05,0.10,100,0.32;1.14,0.30,260,0.35;1.45,0.08,120,0.20",
      DIASTOLIC: "0.05,0.10,100,0.30;0.45,0.08,120,0.22;0.54,0.44,190,0.35;1.05,0.10,100,0.30;1.45,0.08,120,0.22;1.54,0.40,190,0.35",
      S3: "0.05,0.10,100,0.34;0.45,0.08,120,0.22;0.56,0.08,45,0.35;1.05,0.10,100,0.34;1.45,0.08,120,0.22;1.56,0.08,45,0.35"
    };

    function post(path) {
      fetch(path, { method: "POST" }).catch(() => {});
    }

    $("start").onclick = () => post("/api/start");
    $("pause").onclick = () => post("/api/pause");
    $("playAudio").onclick = () => playLatestAudio();

    function render(s) {
      Object.assign(state, s);
      $("dot").className = "dot" + (s.connected ? " on" : "");
      $("status").textContent = `${s.status}${s.running ? " | running" : " | paused"}`;
      $("start").disabled = !s.connected || s.running;
      $("pause").disabled = !s.connected || !s.running;

      const latest = s.latest;
      $("latestClass").textContent = latest ? latest.class : "-";
      $("confidence").textContent = latest ? `${latest.confidence}%` : "-";
      $("latency").textContent = latest ? `${latest.latency} ms` : "-";
      $("input").textContent = latest ? latest.input : "-";
      $("playAudio").disabled = !latest;
      if (latest && latest.input !== renderedInput) {
        renderSignal(latest.input);
      }

      const maxCount = Math.max(1, ...s.classes.map(c => s.counts[c] || 0));
      $("bars").innerHTML = s.classes.map(c => {
        const n = s.counts[c] || 0;
        const pct = Math.round((n / maxCount) * 100);
        return `<div class="bar-row"><div>${c}</div><div class="track"><div class="fill" style="width:${pct}%"></div></div><div>${n}</div></div>`;
      }).join("");

      $("history").innerHTML = s.history.slice(-12).reverse().map(item => {
        const time = new Date(item.ts * 1000).toLocaleTimeString();
        return `<tr><td>${time}</td><td>${item.input}</td><td>${item.class}</td><td>${item.confidence}%</td><td>${item.latency} ms</td></tr>`;
      }).join("");

      $("log").innerHTML = s.logs.slice(-80).map(l => `<div>${escapeHtml(l.line)}</div>`).join("");
      $("log").scrollTop = $("log").scrollHeight;
    }

    function escapeHtml(str) {
      return String(str).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
    }

    function parseEvents(input) {
      const script = synthScripts[input] || synthScripts.NORMAL;
      return script.split(";").map(part => {
        const [start, duration, freq, amp] = part.split(",").map(Number);
        return { start, duration, freq, amp };
      });
    }

    function triangleWave(phase) {
      const frac = phase - Math.floor(phase);
      return frac < 0.5 ? (4 * frac - 1) : (3 - 4 * frac);
    }

    function synthesize(input) {
      const samples = new Float32Array(SAMPLE_RATE * WINDOW_SECONDS);
      const events = parseEvents(input);
      for (const event of events) {
        const start = Math.max(0, Math.floor(event.start * SAMPLE_RATE));
        const count = Math.max(1, Math.floor(event.duration * SAMPLE_RATE));
        const stop = Math.min(samples.length, start + count);
        for (let i = start; i < stop; i++) {
          const pos = (i - start) / count;
          const env = pos < 0.5 ? pos * 2 : (1 - pos) * 2;
          const t = i / SAMPLE_RATE;
          samples[i] += event.amp * env * triangleWave(t * event.freq);
        }
      }
      return { samples, events };
    }

    function renderSignal(input) {
      const data = synthesize(input);
      latestSamples = data.samples;
      renderedInput = input;
      drawWaveform(data.samples);
      drawSpectrogramApprox(data.events);
    }

    function drawWaveform(samples) {
      const canvas = $("waveform");
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#fbfcfe";
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "#d7dde5";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();

      ctx.strokeStyle = "#1677ff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let x = 0; x < w; x++) {
        const start = Math.floor((x / w) * samples.length);
        const end = Math.max(start + 1, Math.floor(((x + 1) / w) * samples.length));
        let peak = 0;
        for (let i = start; i < end; i++) peak = Math.max(peak, Math.abs(samples[i]));
        const yTop = h / 2 - peak * h * 0.42;
        const yBottom = h / 2 + peak * h * 0.42;
        if (x === 0) ctx.moveTo(x, yTop);
        ctx.lineTo(x, yTop);
        ctx.lineTo(x, yBottom);
      }
      ctx.stroke();
    }

    function drawSpectrogramApprox(events) {
      const canvas = $("spectrogram");
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;
      const cols = 64;
      const rows = 64;
      const cellW = w / cols;
      const cellH = h / rows;

      ctx.clearRect(0, 0, w, h);
      for (let x = 0; x < cols; x++) {
        const t = (x / (cols - 1)) * WINDOW_SECONDS;
        for (let y = 0; y < rows; y++) {
          const freq = 25 + Math.pow(y / (rows - 1), 1.8) * 1975;
          let energy = 0.002;
          for (const event of events) {
            const active = t >= event.start && t <= event.start + event.duration;
            if (!active) continue;
            const pos = (t - event.start) / event.duration;
            const env = pos < 0.5 ? pos * 2 : (1 - pos) * 2;
            const bw = Math.max(35, event.freq * 0.35);
            const dist = (freq - event.freq) / bw;
            energy += event.amp * env * Math.exp(-dist * dist);
          }
          const v = Math.min(1, Math.log10(1 + energy * 18));
          const color = spectroColor(v);
          ctx.fillStyle = color;
          ctx.fillRect(x * cellW, h - (y + 1) * cellH, Math.ceil(cellW), Math.ceil(cellH));
        }
      }
    }

    function spectroColor(v) {
      const r = Math.round(20 + 220 * Math.max(0, v - 0.38) / 0.62);
      const g = Math.round(34 + 160 * Math.sin(v * Math.PI));
      const b = Math.round(78 + 145 * (1 - v));
      return `rgb(${r},${g},${b})`;
    }

    function playLatestAudio() {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const buffer = ctx.createBuffer(1, latestSamples.length, SAMPLE_RATE);
      buffer.copyToChannel(latestSamples, 0);
      const source = ctx.createBufferSource();
      const gain = ctx.createGain();
      gain.gain.value = 0.7;
      source.buffer = buffer;
      source.connect(gain);
      gain.connect(ctx.destination);
      source.start();
    }

    renderSignal("NORMAL");

    const events = new EventSource("/events");
    events.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.state) render(msg.state);
      if (msg.type === "log" && !msg.state) {
        state.logs = (state.logs || []).concat([{ ts: msg.ts, line: msg.line }]).slice(-80);
        render(state);
      }
    };
  </script>
</body>
</html>
"""


def make_handler(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML.encode("utf-8"))
                return

            if parsed.path == "/events":
                q = state.register()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    while True:
                        msg = q.get(timeout=20)
                        self.wfile.write(msg.encode("utf-8"))
                        self.wfile.flush()
                except Exception:
                    state.unregister(q)
                return

            if parsed.path == "/api/state":
                self.send_json(state.snapshot())
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/start":
                ok = state.command(b"S")
                self.send_json({"ok": ok, "state": state.snapshot()})
                return
            if parsed.path == "/api/pause":
                ok = state.command(b"P")
                self.send_json({"ok": ok, "state": state.snapshot()})
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def send_json(self, payload: dict):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):  # keep console readable
            return

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8765)
    args = parser.parse_args()

    state = DashboardState(args.port, args.baud)
    state.start_reader()

    server = ThreadingHTTPServer((args.host, args.http_port), make_handler(state))
    print(f"Dashboard: http://{args.host}:{args.http_port}")
    print(f"Serial   : {args.port} @ {args.baud}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop_event.set()
        with state.serial_lock:
            if state.serial is not None:
                state.serial.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
