"""
03_quantize.py — Train Keras ResNet-10, then INT8 post-training quantization

Flow:
  1. Build Keras ResNet-10 (same architecture as models/resnet10.py)
  2. Train on PhysioNet mel-spectrograms (or load existing SavedModel)
  3. INT8 PTQ via TFLite converter with representative calibration dataset
  4. Validate: accuracy drop < 2% vs float32

Usage:
    python ml/03_quantize.py [--data-dir ml/data] [--epochs 50] [--calib-clips 200]
    python ml/03_quantize.py --skip-train   # reuse existing SavedModel

Output:
    ml/models/resnet10_int8.tflite
    ml/models/resnet10_float32.tflite
    ml/models/resnet10_savedmodel/
    ml/models/keras_best.keras
"""

import argparse
import numpy as np
from pathlib import Path

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")


# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["normal", "systolic", "diastolic", "s3gallop"]
N_CLASSES   = 4
IMG_SIZE    = 64


# ── Model definition ──────────────────────────────────────────────────────────

def _se_block(x, channels, reduction=4, name=""):
    # Use keepdims=False + explicit Reshape to avoid SHAPE/GATHER/PACK/REDUCE_PROD
    # ops that TFLite Micro does not support (caused by keepdims=True + Dense).
    gap = tf.keras.layers.GlobalAveragePooling2D(keepdims=False, name=f"{name}_gap")(x)
    fc1 = tf.keras.layers.Dense(channels // reduction, activation='relu',
                                  name=f"{name}_fc1")(gap)
    fc2 = tf.keras.layers.Dense(channels, activation='sigmoid',
                                  name=f"{name}_fc2")(fc1)
    fc2 = tf.keras.layers.Reshape((1, 1, channels), name=f"{name}_reshape")(fc2)
    return tf.keras.layers.Multiply(name=f"{name}_mul")([x, fc2])


def _res_block(x, out_ch, stride=1, use_se=True, name=""):
    shortcut = x

    x = tf.keras.layers.Conv2D(out_ch, 3, strides=stride, padding='same',
                                use_bias=False, name=f"{name}_c1")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = tf.keras.layers.ReLU(name=f"{name}_relu1")(x)
    x = tf.keras.layers.Conv2D(out_ch, 3, strides=1, padding='same',
                                use_bias=False, name=f"{name}_c2")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)

    if use_se:
        x = _se_block(x, out_ch, name=f"{name}_se")

    in_ch = shortcut.shape[-1]
    if stride != 1 or in_ch != out_ch:
        shortcut = tf.keras.layers.Conv2D(out_ch, 1, strides=stride,
                                           use_bias=False, name=f"{name}_proj")(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f"{name}_projbn")(shortcut)

    x = tf.keras.layers.Add(name=f"{name}_add")([x, shortcut])
    return tf.keras.layers.ReLU(name=f"{name}_relu2")(x)


def build_resnet10(use_se=True):
    """Input: (batch, 64, 64, 1)  ->  Output: (batch, 4) softmax."""
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="input")
    x   = tf.keras.layers.Conv2D(16, 7, strides=2, padding='same',
                                  use_bias=False, name="stem_conv")(inp)
    x   = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x   = tf.keras.layers.ReLU(name="stem_relu")(x)
    x   = tf.keras.layers.MaxPool2D(2, strides=2, name="stem_pool")(x)
    x   = _res_block(x, 16, stride=1, use_se=use_se, name="rb1")
    x   = _res_block(x, 32, stride=2, use_se=use_se, name="rb2")
    x   = _res_block(x, 64, stride=2, use_se=use_se, name="rb3")
    x   = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    out = tf.keras.layers.Dense(N_CLASSES, activation='softmax', name="fc")(x)
    return tf.keras.Model(inputs=inp, outputs=out, name="resnet10")


# ── Data helpers ──────────────────────────────────────────────────────────────

def collect_paths(processed_dir: Path, split: str):
    """Returns (paths: list[str], labels: list[int])."""
    paths, labels = [], []
    for idx, name in enumerate(CLASS_NAMES):
        d = processed_dir / split / name
        if not d.exists():
            continue
        for f in sorted(d.glob("*.npy")):
            paths.append(str(f))
            labels.append(idx)
    return paths, labels


def make_dataset(paths, labels, norm_mean, norm_std,
                 augment=False, batch_size=32):
    """Build a tf.data.Dataset from .npy paths + integer labels."""
    def load_fn(path, label):
        spec = tf.numpy_function(
            lambda p: np.load(p.decode()).astype(np.float32), [path], tf.float32)
        spec.set_shape([IMG_SIZE, IMG_SIZE])
        spec = (spec - norm_mean) / (norm_std + 1e-8)
        spec = tf.expand_dims(spec, -1)          # -> (64, 64, 1)
        return spec, label

    def augment_fn(spec, label):
        # Circular time-shift ±6 frames
        shift = tf.random.uniform([], -6, 7, dtype=tf.int32)
        spec  = tf.roll(spec, shift, axis=1)
        # Frequency mask (2–4 bands)
        idx   = tf.range(IMG_SIZE)
        f0    = tf.random.uniform([], 0, 60, dtype=tf.int32)
        fm    = tf.random.uniform([], 2, 5,  dtype=tf.int32)
        mf    = tf.cast(tf.logical_or(idx < f0, idx >= f0 + fm), tf.float32)
        spec  = spec * mf[:, tf.newaxis, tf.newaxis]
        # Time mask (5–10 frames)
        t0    = tf.random.uniform([], 0, 54, dtype=tf.int32)
        tm    = tf.random.uniform([], 5, 11, dtype=tf.int32)
        mt    = tf.cast(tf.logical_or(idx < t0, idx >= t0 + tm), tf.float32)
        spec  = spec * mt[tf.newaxis, :, tf.newaxis]
        # Gaussian noise
        spec  = spec + tf.random.normal(tf.shape(spec), stddev=0.05)
        return spec, label

    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(paths), tf.constant(labels, dtype=tf.int32)))
    if augment:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)
    ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_weights_from_labels(labels):
    counts  = np.bincount(labels, minlength=N_CLASSES).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * N_CLASSES
    return {i: float(w) for i, w in enumerate(weights)}


def load_calibration_data(processed_dir, n_clips, norm_mean, norm_std):
    specs = []
    per_class = max(1, n_clips // N_CLASSES)
    for name in CLASS_NAMES:
        d = processed_dir / "train" / name
        if not d.exists():
            continue
        for f in sorted(d.glob("*.npy"))[:per_class]:
            s = np.load(f).astype(np.float32)
            specs.append((s - norm_mean) / (norm_std + 1e-8))
    arr = np.stack(specs[:n_clips], 0)
    return arr[:, :, :, np.newaxis]   # (N, 64, 64, 1)


# ── TFLite accuracy helper ────────────────────────────────────────────────────

def eval_tflite(model_bytes, test_specs, test_labels):
    interp = tf.lite.Interpreter(model_content=model_bytes)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    preds = []
    for spec in test_specs:
        inp = spec[np.newaxis]
        if inp_d['dtype'] == np.int8:
            scale, zp = inp_d['quantization']
            inp = (inp / scale + zp).clip(-128, 127).astype(np.int8)
        interp.set_tensor(inp_d['index'], inp)
        interp.invoke()
        out = interp.get_tensor(out_d['index'])
        if out_d['dtype'] == np.int8:
            scale, zp = out_d['quantization']
            out = (out.astype(np.float32) - zp) * scale
        preds.append(int(np.argmax(out)))

    preds = np.array(preds)
    acc   = (preds == test_labels).mean()
    return acc, preds


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    default="ml/data")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--calib-clips", type=int,   default=200)
    parser.add_argument("--no-se",       action="store_true")
    parser.add_argument("--skip-train",  action="store_true",
                        help="Skip training, load existing SavedModel")
    parser.add_argument("--saved-model", default="ml/models/resnet10_savedmodel")
    args = parser.parse_args()

    data_root  = Path(args.data_dir)
    processed  = data_root / "processed"
    model_dir  = Path("ml/models")
    saved_path = Path(args.saved_model)
    norm_path  = data_root / "normalization_params.npy"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Normalization params
    if norm_path.exists():
        p = np.load(norm_path)
        norm_mean, norm_std = float(p[0]), float(p[1])
        print(f"Normalization: mean={norm_mean:.4f}, std={norm_std:.4f}")
    else:
        norm_mean, norm_std = 0.0, 1.0
        print("WARNING: normalization_params.npy not found — using mean=0, std=1")

    ckpt_path = model_dir / "keras_best.keras"

    # ── Build or load model ───────────────────────────────────────────────────
    if args.skip_train and saved_path.exists():
        print(f"\nLoading SavedModel from {saved_path}...")
        keras_model = tf.keras.models.load_model(str(saved_path))
        keras_model.summary(line_length=90)

    elif args.skip_train and ckpt_path.exists():
        # SavedModel wasn't persisted yet (e.g. prior crash) but keras checkpoint is
        print(f"\nLoading Keras checkpoint from {ckpt_path} and saving SavedModel...")
        keras_model = tf.keras.models.load_model(str(ckpt_path))
        keras_model.summary(line_length=90)
        keras_model.save(str(saved_path))
        print(f"SavedModel saved -> {saved_path}")

    else:
        print("\n=== Building Keras ResNet-10 ===")
        keras_model = build_resnet10(use_se=not args.no_se)
        keras_model.summary(line_length=90)
        print(f"Total params: {keras_model.count_params():,}")

        # Collect paths
        train_paths, train_lbls = collect_paths(processed, "train")
        val_paths,   val_lbls   = collect_paths(processed, "val")

        if not train_paths:
            raise FileNotFoundError(
                f"No training data in {processed}/train. Run 01_preprocess.py first.")

        print(f"\n=== Dataset ===")
        print(f"  Train: {len(train_paths)}  Val: {len(val_paths)}")
        for i, name in enumerate(CLASS_NAMES):
            tc = sum(1 for l in train_lbls if l == i)
            vc = sum(1 for l in val_lbls   if l == i)
            print(f"    {name}: train={tc}, val={vc}")

        class_weights = class_weights_from_labels(np.array(train_lbls))
        print(f"  Class weights: {class_weights}")

        train_ds = make_dataset(train_paths, train_lbls, norm_mean, norm_std,
                                augment=True,  batch_size=args.batch_size)
        val_ds   = make_dataset(val_paths,   val_lbls,   norm_mean, norm_std,
                                augment=False, batch_size=args.batch_size)

        # Cosine LR decay
        steps_per_epoch = max(1, len(train_paths) // args.batch_size)
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.lr,
            decay_steps=args.epochs * steps_per_epoch,
            alpha=1e-5 / args.lr
        )

        keras_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule,
                                                weight_decay=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(ckpt_path), save_best_only=True,
                monitor='val_accuracy', mode='max', verbose=1),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=10,
                restore_best_weights=True, verbose=1),
        ]

        print(f"\n=== Training ({args.epochs} epochs, target val acc >80%) ===")
        history = keras_model.fit(
            train_ds, validation_data=val_ds,
            epochs=args.epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        best_val_acc = max(history.history.get('val_accuracy', [0.0]))
        print(f"\nBest val accuracy: {best_val_acc*100:.1f}%")
        if best_val_acc < 0.70:
            print("WARNING: val accuracy < 70% — check data quality and label distribution")

        # Reload best checkpoint
        if ckpt_path.exists():
            keras_model = tf.keras.models.load_model(str(ckpt_path))
            print(f"Loaded best checkpoint: {ckpt_path}")

        print(f"\nSaving SavedModel -> {saved_path}")
        keras_model.save(str(saved_path))

    # ── Test set evaluation (Keras) ───────────────────────────────────────────
    test_paths, test_lbls = collect_paths(processed, "test")
    if test_paths:
        test_ds = make_dataset(test_paths, test_lbls, norm_mean, norm_std,
                               augment=False, batch_size=args.batch_size)
        _, keras_test_acc = keras_model.evaluate(test_ds, verbose=0)
        print(f"\nKeras float32 test accuracy: {keras_test_acc*100:.1f}%")

    # ── Float32 TFLite ────────────────────────────────────────────────────────
    print("\n=== Converting to float32 TFLite ===")
    conv_f32   = tf.lite.TFLiteConverter.from_saved_model(str(saved_path))
    tflite_f32 = conv_f32.convert()
    f32_path   = model_dir / "resnet10_float32.tflite"
    f32_path.write_bytes(tflite_f32)
    print(f"  {f32_path}  ({len(tflite_f32)/1024:.1f} KB)")

    # ── INT8 PTQ ──────────────────────────────────────────────────────────────
    print(f"\n=== INT8 PTQ (calib clips: {args.calib_clips}) ===")
    calib = load_calibration_data(processed, args.calib_clips, norm_mean, norm_std)
    print(f"  Calibration shape: {calib.shape}")

    def representative_dataset():
        for i in range(len(calib)):
            yield [calib[i:i+1]]

    conv_int8 = tf.lite.TFLiteConverter.from_saved_model(str(saved_path))
    conv_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_int8.representative_dataset = representative_dataset
    conv_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv_int8.inference_input_type  = tf.int8
    conv_int8.inference_output_type = tf.int8

    tflite_int8 = conv_int8.convert()
    int8_path   = model_dir / "resnet10_int8.tflite"
    int8_path.write_bytes(tflite_int8)
    print(f"  {int8_path}  ({len(tflite_int8)/1024:.1f} KB)")
    print(f"  Compression: {len(tflite_f32)/len(tflite_int8):.1f}x vs float32")

    # ── Accuracy validation ───────────────────────────────────────────────────
    if test_paths:
        print("\n=== Quantization accuracy validation ===")
        test_specs = np.stack([
            (np.load(p).astype(np.float32) - norm_mean) / (norm_std + 1e-8)
            for p in test_paths
        ])[:, :, :, np.newaxis]
        test_lbls_arr = np.array(test_lbls)

        acc_f32,  _ = eval_tflite(tflite_f32,  test_specs, test_lbls_arr)
        acc_int8, _ = eval_tflite(tflite_int8, test_specs, test_lbls_arr)
        drop = acc_f32 - acc_int8

        print(f"  Float32 TFLite: {acc_f32*100:.1f}%")
        print(f"  INT8    TFLite: {acc_int8*100:.1f}%")
        print(f"  Drop:           {drop*100:.2f}%  "
              f"{'PASS (<2%)' if drop < 0.02 else 'FAIL (>2%) — increase --calib-clips'}")

    print(f"\n=== Done ===")
    print(f"  INT8 model: {int8_path}")
    print("Next: python ml/04_export.py")


if __name__ == "__main__":
    main()
