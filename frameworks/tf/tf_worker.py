"""TensorFlow lowering worker (runs in its own process).

TF and jaxlib each bundle their own MLIR/LLVM; co-loading them live segfaults.
So TF runs here, isolated: it XLA-compiles a tf.function, dumps the StableHLO
text + the exact input arrays + TF's own reference output to a directory. The
driver (test_tf_e2e.py, TF-free) then parses that StableHLO and runs it on VGRE.

Usage:  python tf_worker.py <case-name> <out-dir>
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf


def _r(*shape):
    return np.random.default_rng(abs(hash(shape)) % (2**31)).standard_normal(shape).astype(np.float32)


# Each case: name -> (tf-callable, list of input ndarrays). jit_compile=True forces XLA.
def cases():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3)).astype(np.float32)
    W = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((4,)).astype(np.float32)
    img = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)   # NCHW
    ker = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)   # OIHW
    return {
        "matmul":        (lambda a, w: tf.matmul(a, w), [x, W]),
        "dense_relu":    (lambda a, w, c: tf.nn.relu(tf.matmul(a, w) + c), [x, W, b]),
        "add_mul":       (lambda a, c: a * c + a, [x, x]),
        "tanh":          (lambda a, w, c: tf.tanh(tf.matmul(a, w) + c), [x, W, b]),
        "sigmoid":       (lambda a: tf.nn.sigmoid(a), [x]),
        "softmax":       (lambda a: tf.nn.softmax(a, axis=-1), [x]),
        "reduce_sum":    (lambda a: tf.reduce_sum(a, axis=1), [x]),
        "reduce_mean":   (lambda a: tf.reduce_mean(a, axis=1), [x]),
        "reduce_max":    (lambda a: tf.reduce_max(a, axis=0), [x]),
        "transpose":     (lambda a: tf.transpose(a), [x]),
        "exp_log_abs":   (lambda a: tf.abs(tf.math.log(tf.exp(a) + 2.0)), [x]),
        "rsqrt":         (lambda a: a * tf.math.rsqrt(tf.reduce_sum(a * a, 1, keepdims=True) + 1e-6), [x]),
        "concat":        (lambda a, c: tf.concat([a, c], axis=1), [x, x]),
        "relu_conv2d":   (lambda im, kr: tf.nn.relu(tf.nn.conv2d(
                              im, tf.transpose(kr, [2, 3, 1, 0]), strides=[1, 1, 1, 1],
                              padding="SAME", data_format="NCHW")), [img, ker]),
        "maxpool2d":     (lambda im: tf.nn.max_pool2d(im, ksize=2, strides=2,
                              padding="VALID", data_format="NCHW"), [img]),
        "avgpool2d":     (lambda im: tf.nn.avg_pool2d(im, ksize=2, strides=2,
                              padding="VALID", data_format="NCHW"), [img]),
    }


def main():
    name, out_dir = sys.argv[1], sys.argv[2]
    fn, inputs = cases()[name]
    os.makedirs(out_dir, exist_ok=True)
    g = tf.function(fn, jit_compile=True)
    targs = [tf.constant(a) for a in inputs]
    txt = g.experimental_get_compiler_ir(*targs)(stage="stablehlo")
    with open(os.path.join(out_dir, "m.mlir"), "w") as f:
        f.write(txt)
    for i, a in enumerate(inputs):
        np.save(os.path.join(out_dir, f"in_{i}.npy"), a)
    ref = np.asarray(g(*targs), np.float32).ravel()
    np.save(os.path.join(out_dir, "ref.npy"), ref)
    print(f"dumped {name}: {len(inputs)} inputs, ref size {ref.size}")


if __name__ == "__main__":
    main()
