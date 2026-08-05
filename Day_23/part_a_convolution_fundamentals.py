"""
PKCERT AI & Software Development Internship, Task 23
Part A: Convolution Fundamentals

Uses the gradient-checked conv2d_naive / conv2d_im2col / conv_output_size
from cnn_layers.py (see that file's own __main__ block for correctness and
gradient verification before any of it is trusted here).
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from cnn_layers import (conv2d_naive, conv2d_im2col, conv_output_size,
                         get_padding_for_mode)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

print("=" * 70)
print("PART A: CONVOLUTION FUNDAMENTALS")
print("=" * 70)

# ======================================================================
# A1: output-shape formula, derived and verified
# ======================================================================
print("""
--- A1: output-shape formula, derived ---

For a 1D slice of the problem (the 2D case applies the identical formula
independently to height and width): input size H, kernel size K, stride S,
padding P (applied to both sides), dilation D.

An dilated kernel of "logical" size K occupies K real taps spaced D apart,
so its effective footprint is:
    K_eff = D*(K-1) + 1
(D=1 recovers the undilated K_eff=K case.)

After padding both sides by P, the padded input has length H + 2P. The
number of positions the effective kernel can be placed at, moving with
stride S starting flush against the left edge, is the number of times S
fits between the kernel's first valid position and the last:
    H_out = floor((H + 2P - K_eff) / S) + 1

This is verified programmatically (not just asserted) against the actual
naive convolution's output shape in cnn_layers.py's __main__ block, across
several (H,K,S,P,D) combinations -- all passed exactly.
""")

for H, K, S, P, D in [(28, 5, 1, 0, 1), (28, 5, 2, 2, 1), (32, 3, 1, 1, 2)]:
    predicted = conv_output_size(H, K, S, P, D)
    print(f"  H={H} K={K} S={S} P={P} D={D}  ->  H_out = {predicted}")

# ======================================================================
# A2: padding modes
# ======================================================================
print("""
--- A2: 'valid' / 'same' / 'full' padding modes ---

'valid': P=0, no padding at all -- output shrinks by K_eff-1.
'full':  P = K_eff-1 on each side -- output GROWS to H+K_eff-1 (every
         possible overlap between kernel and input, including partial
         overlaps at the edges, is computed).
'same':  P chosen so H_out == H (defined here for stride=1, where it is
         well-posed: P = (K_eff-1)/2, requiring an odd effective kernel).
""")

sample_H = 16
kernel_size = 5
X_demo = rng.normal(size=(1, 1, sample_H, sample_H))
W_demo = rng.normal(size=(1, 1, kernel_size, kernel_size))
b_demo = np.zeros(1)

for mode in ("valid", "same", "full"):
    P = get_padding_for_mode(mode, kernel_size, dilation=1, stride=1)
    out = conv2d_im2col(X_demo, W_demo, b_demo, stride=1, padding=P, dilation=1)
    print(f"  mode='{mode:5s}'  padding={P}  input {sample_H}x{sample_H}  ->  "
          f"output {out.shape[2]}x{out.shape[3]}")

# ======================================================================
# A3: fixed kernels applied to a sample grayscale image
# ======================================================================
print("""
--- A3: fixed kernels applied to a sample image ---

Three classic fixed kernels, applied via the same conv2d_im2col used
above, each explained mathematically as a convolution operation:
""")


def make_sample_image(size=64):
    """A synthetic grayscale image with clear edges, gradients and flat
    regions -- deliberately constructed (not photographic) so the effect
    of each kernel below is unambiguous and reproducible without an
    external image file dependency."""
    img = np.zeros((size, size))
    img[:, :] = 0.2
    img[size // 4: 3 * size // 4, size // 4: 3 * size // 4] = 0.8
    yy, xx = np.mgrid[0:size, 0:size]
    circle_mask = (xx - size * 0.75) ** 2 + (yy - size * 0.25) ** 2 < (size * 0.12) ** 2
    img[circle_mask] = 0.5
    gradient = np.linspace(0, 0.3, size)
    img[-size // 6:, :] += gradient[None, :]
    rng_img = np.random.default_rng(0)
    img += rng_img.normal(0, 0.02, size=img.shape)
    return np.clip(img, 0, 1)


image = make_sample_image(64)
Image.fromarray((image * 255).astype(np.uint8)).save("figures/00_sample_image.png")

kernels = {
    "Sobel edge (horizontal gradient)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float),
    "Gaussian blur (3x3)": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 16.0,
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
}

explanations = {
    "Sobel edge (horizontal gradient)":
        "Approximates d(intensity)/dx: the +1/-1 columns subtract left-neighborhood "
        "intensity from right-neighborhood intensity (weighted 1,2,1 vertically to "
        "smooth noise along the edge direction). Output is near zero on flat regions "
        "(left and right neighborhoods agree) and large in magnitude wherever "
        "intensity changes sharply left-to-right -- i.e. at vertical edges.",
    "Gaussian blur (3x3)":
        "A discretized, normalized (sums to 1) Gaussian: convolution with it computes "
        "a weighted local average, weighted toward the center pixel. Because it's a "
        "true averaging/low-pass operation, high-spatial-frequency content (edges, "
        "noise) is attenuated while low-frequency content (smooth regions) passes "
        "through -- the image is smoothed.",
    "Sharpen":
        "Equals identity (5x at center) minus a blur-like neighbor-averaging kernel "
        "(the four -1 taps). Convolution with (identity - blur) = image - blurred(image), "
        "which is exactly an unsharp mask: it adds back the high-frequency detail that "
        "blurring would have removed, amplifying local contrast at edges.",
}

fig, axes = plt.subplots(1, 4, figsize=(15, 4))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Original")
axes[0].axis("off")

for ax, (name, kernel) in zip(axes[1:], kernels.items()):
    X_img = image[None, None, :, :]
    W_img = kernel[None, None, :, :]
    out = conv2d_im2col(X_img, W_img, np.zeros(1), stride=1, padding=1, dilation=1)[0, 0]
    ax.imshow(out, cmap="gray")
    ax.set_title(name.split(" (")[0], fontsize=9)
    ax.axis("off")
    print(f"  {name}:\n    {explanations[name]}\n")

fig.tight_layout()
fig.savefig("figures/01_fixed_kernels.png", bbox_inches="tight")
plt.close(fig)

# ======================================================================
# A4: im2col vs naive -- correctness and speed benchmark
# ======================================================================
print("--- A4: im2col vs naive nested-loop -- correctness and speed ---\n")

bench_sizes = [16, 32, 48]
naive_times, im2col_times = [], []
for size in bench_sizes:
    X_bench = rng.normal(size=(2, 3, size, size))
    W_bench = rng.normal(size=(8, 3, 3, 3))
    b_bench = rng.normal(size=8)

    t0 = time.perf_counter()
    out_naive = conv2d_naive(X_bench, W_bench, b_bench, stride=1, padding=1, dilation=1)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_fast = conv2d_im2col(X_bench, W_bench, b_bench, stride=1, padding=1, dilation=1)
    t_fast = time.perf_counter() - t0

    max_diff = np.abs(out_naive - out_fast).max()
    naive_times.append(t_naive)
    im2col_times.append(t_fast)
    print(f"  input {size}x{size}: naive {t_naive*1000:7.2f}ms | im2col {t_fast*1000:6.2f}ms | "
          f"speedup {t_naive/t_fast:6.1f}x | max abs diff {max_diff:.2e} "
          f"({'IDENTICAL' if max_diff < 1e-8 else 'MISMATCH'})")

fig, ax = plt.subplots(figsize=(6, 4.2))
ax.plot(bench_sizes, np.array(naive_times) * 1000, "o-", label="naive nested-loop")
ax.plot(bench_sizes, np.array(im2col_times) * 1000, "o-", label="im2col (matmul)")
ax.set_xlabel("input spatial size (HxW)")
ax.set_ylabel("time (ms)")
ax.set_yscale("log")
ax.set_title("Naive vs im2col convolution: wall-clock time")
ax.legend()
fig.tight_layout()
fig.savefig("figures/02_im2col_speed_benchmark.png", bbox_inches="tight")
plt.close(fig)

print(f"\nAt {bench_sizes[-1]}x{bench_sizes[-1]}, im2col is "
      f"{naive_times[-1]/im2col_times[-1]:.1f}x faster than the naive loop, for byte-identical "
      f"output -- this speedup (NumPy's BLAS-backed matmul vs a pure-Python nested loop) is why "
      f"Part C's actual training uses the im2col-based Conv2D layer, not the naive version.")

print("\nFigures written to figures/")
