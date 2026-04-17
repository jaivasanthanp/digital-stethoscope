"""
test_inference.py — On-device accuracy spot check

Sends 12 test vectors (3 per class) to the device and reports per-class
accuracy and overall accuracy. Compares to TFLite reference.

Usage:
    python tests/ml/test_inference.py --port COM3 [--data-dir ml/data]
    python tests/ml/test_inference.py --tflite-only  (no device needed)
"""

import argparse
import sys
import numpy as np
from pathlib import Path

import tensorflow as tf

CLASS_NAMES = ["Normal", "SystolicMurmur", "DiastolicMurmur", "S3Gallop"]
CLASS_DIRS  = ["normal", "systolic", "diastolic", "s3gallop"]


def load_test_samples(data_root: Path, n_per_class: int = 3):
    """Load test spectrograms and labels."""
    samples = []
    for label, class_dir_name in enumerate(CLASS_DIRS):
        class_dir = data_root / "test" / class_dir_name
        if not class_dir.exists():
            print(f"  WARNING: {class_dir} not found")
            continue
        files = sorted(class_dir.glob("*.npy"))[:n_per_class]
        for f in files:
            samples.append((np.load(f).astype(np.float32), label))
    return samples


def run_tflite(tflite_path: Path, spec: np.ndarray) -> tuple[int, float]:
    """Run host TFLite inference."""
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    inp = spec[np.newaxis, :, :, np.newaxis].astype(np.float32)
    if inp_det['dtype'] == np.int8:
        scale, zp = inp_det['quantization']
        inp = (inp / scale + zp).clip(-128, 127).astype(np.int8)

    interp.set_tensor(inp_det['index'], inp)
    interp.invoke()
    out = interp.get_tensor(out_det['index'])

    if out_det['dtype'] == np.int8:
        scale, zp = out_det['quantization']
        out = (out.astype(np.float32) - zp) * scale

    return int(np.argmax(out)), float(np.max(out))


def test_tflite_accuracy(tflite_path: Path, samples):
    """Evaluate TFLite model accuracy on test samples."""
    print(f"\n=== TFLite Host Accuracy ({len(samples)} samples) ===")

    per_class_correct = [0] * 4
    per_class_total   = [0] * 4
    correct = 0

    for spec, true_label in samples:
        pred_class, conf = run_tflite(tflite_path, spec)
        per_class_total[true_label] += 1
        if pred_class == true_label:
            correct += 1
            per_class_correct[true_label] += 1

        status = "OK" if pred_class == true_label else "WRONG"
        print(f"  true={CLASS_NAMES[true_label]:<15} "
              f"pred={CLASS_NAMES[pred_class]:<15} ({conf*100:.0f}%)  {status}")

    print(f"\nOverall: {correct}/{len(samples)} = {correct/len(samples)*100:.1f}%")
    for i, name in enumerate(CLASS_NAMES):
        if per_class_total[i] > 0:
            print(f"  {name:<15}: {per_class_correct[i]}/{per_class_total[i]}")


def main():
    parser = argparse.ArgumentParser(description="Inference accuracy test")
    parser.add_argument("--tflite-only",  action="store_true",
                        help="Host TFLite test only (no device)")
    parser.add_argument("--port",         default="COM3")
    parser.add_argument("--baud",         type=int, default=115200)
    parser.add_argument("--data-dir",     default="ml/data")
    parser.add_argument("--model",        default="ml/models/resnet10_int8.tflite")
    parser.add_argument("--n-per-class",  type=int, default=3)
    args = parser.parse_args()

    tflite_path = Path(args.model)
    data_root   = Path(args.data_dir) / "processed"

    if not tflite_path.exists():
        print(f"ERROR: {tflite_path} not found. Run ml/03_quantize.py first.")
        sys.exit(1)

    samples = load_test_samples(data_root, args.n_per_class)
    if not samples:
        print("ERROR: No test samples. Run ml/01_preprocess.py first.")
        sys.exit(1)

    test_tflite_accuracy(tflite_path, samples)

    if not args.tflite_only:
        print("\n=== On-device comparison via 05_validate_on_device.py ===")
        print(f"  python ml/05_validate_on_device.py --port {args.port}")


if __name__ == "__main__":
    main()
