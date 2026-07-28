"""
PKCERT AI & Software Development Internship, Task 18
Part A: Tensors & Basic Operations

Installation check, four ways to create a tensor, indexing/slicing/reshape/
broadcasting, a timed matmul against NumPy, and the torch.Tensor <-> NumPy
ndarray relationship.
"""

import time
import numpy as np
import torch

print(f"=== Part A: PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()} ===")
print("(CPU is used throughout this submission; no GPU is required or assumed.)")

# ----------------------------------------------------------------------
# A2: four ways to create a tensor
# ----------------------------------------------------------------------
print("\n--- Four ways to create a tensor ---")

t_from_list = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"torch.tensor(nested list): use when you already have concrete values in "
      f"Python/list form (constants, small hand-written examples).\n{t_from_list}")

t_zeros = torch.zeros(2, 3)
t_ones = torch.ones(2, 3)
print(f"\ntorch.zeros/torch.ones: use for placeholders or accumulators of a known "
      f"shape (e.g. initial biases at 0, a running-sum buffer, masks).\n{t_zeros}\n{t_ones}")

t_arange = torch.arange(0, 10, 2)
print(f"\ntorch.arange: use for index ranges, positional encodings, or evenly spaced "
      f"schedules (e.g. epoch numbers, x-axis coordinates).\n{t_arange}")

t_rand = torch.rand(2, 3)
print(f"\ntorch.rand: use for weight initialisation and stochastic sampling -- values "
      f"uniform in [0, 1).\n{t_rand}")

np_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
t_from_numpy = torch.from_numpy(np_array)
print(f"\ntorch.from_numpy: use when data already lives in NumPy (e.g. loaded with "
      f"pandas/NumPy) and needs to enter a PyTorch model without copying it.\n{t_from_numpy}")

# ----------------------------------------------------------------------
# A3: indexing, slicing, reshaping, broadcasting
# ----------------------------------------------------------------------
print("\n--- Indexing, slicing, reshaping, broadcasting ---")
x = torch.arange(24).reshape(4, 6)
print(f"x (4x6):\n{x}")
print(f"x[1, 3] (single element): {x[1, 3].item()}")
print(f"x[:, 2] (column 2): {x[:, 2].tolist()}")
print(f"x[1:3, 2:5] (sub-block): \n{x[1:3, 2:5]}")
print(f"x[::2, :] (every other row): \n{x[::2, :]}")

x_view = x.view(2, 12)
x_reshape = x.reshape(3, 8)
x_view[0, 0] = 999
print(f"\nx.view(2,12) shares memory with x: setting x_view[0,0]=999 also changed "
      f"x[0,0] -> {x[0, 0].item()} (view never copies, requires the tensor to already "
      f"be contiguous in memory)")
print(f"x.reshape(3,8) also (usually) returns a view when possible, but reshape() "
      f"is the safer general-purpose choice: unlike view(), it silently falls back to "
      f"copying the data when the requested shape is not viewable from the current "
      f"memory layout (e.g. after a non-contiguous-producing op like .transpose()), so "
      f"it never raises where view() would.")
x[0, 0] = 0  # restore for the rest of the script

a = torch.ones(4, 1)
b = torch.arange(3).reshape(1, 3)
print(f"\nBroadcasting: a is (4,1), b is (1,3). a + b broadcasts both to (4,3) by "
      f"repeating each along the size-1 dimension, with no data actually copied:")
print(a + b)
try:
    torch.ones(4, 2) + torch.ones(3, 2)
except RuntimeError as e:
    print(f"\nIncompatible shapes (4,2) and (3,2) correctly raise: {e}")

# ----------------------------------------------------------------------
# A4: timed matrix multiplication, NumPy vs PyTorch
# ----------------------------------------------------------------------
print("\n--- Matrix multiplication: NumPy vs PyTorch, timed ---")
N = 1200
rng = np.random.default_rng(42)
A_np = rng.random((N, N), dtype=np.float64)
B_np = rng.random((N, N), dtype=np.float64)
A_t = torch.from_numpy(A_np)
B_t = torch.from_numpy(B_np)

# warm-up (first call pays one-off setup cost in both libraries' BLAS backends)
_ = A_np @ B_np
_ = A_t @ B_t

n_runs = 5
np_times, torch_times = [], []
for _ in range(n_runs):
    t0 = time.perf_counter(); _ = A_np @ B_np; np_times.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); _ = A_t @ B_t; torch_times.append(time.perf_counter() - t0)

np_mean, torch_mean = np.mean(np_times), np.mean(torch_times)
print(f"{N}x{N} @ {N}x{N} matmul, mean of {n_runs} runs:")
print(f"  NumPy:   {np_mean * 1000:.2f} ms")
print(f"  PyTorch: {torch_mean * 1000:.2f} ms")
faster = "PyTorch" if torch_mean < np_mean else "NumPy"
ratio = max(np_mean, torch_mean) / min(np_mean, torch_mean)
print(f"  {faster} was {ratio:.2f}x faster on this CPU. Both ultimately call into an "
      f"optimised BLAS library under the hood for float64 matmul on CPU, so a large gap "
      f"either way usually reflects which BLAS backend each library happened to link "
      f"against on this machine, not an inherent advantage of one library's Python "
      f"layer -- the actual arithmetic is done by the same class of vendor-tuned code "
      f"in both cases.")

# ----------------------------------------------------------------------
# A5: torch.Tensor vs numpy.ndarray
# ----------------------------------------------------------------------
print("\n--- torch.Tensor vs numpy.ndarray: the relationship ---")
shared_np = np.array([1.0, 2.0, 3.0])
shared_t = torch.from_numpy(shared_np)
shared_t[0] = 100.0
print(f"torch.from_numpy() shares the underlying memory buffer, it does not copy: "
      f"after setting shared_t[0]=100, the original NumPy array is also changed -> "
      f"{shared_np}")

back_to_np = shared_t.numpy()
back_to_np[1] = 200.0
print(f".numpy() shares memory the same way, in reverse: after setting "
      f"back_to_np[1]=200, the tensor changed too -> {shared_t}")

independent_t = torch.tensor(shared_np)
independent_t[2] = -1.0
print(f"By contrast, torch.tensor(existing_array) always COPIES: setting "
      f"independent_t[2]=-1 left the original array untouched -> {shared_np}")

print("""
In short: a torch.Tensor and a NumPy ndarray are, underneath the API, the same
kind of object -- a contiguous block of raw memory plus shape/stride/dtype
metadata describing how to read it. torch.from_numpy() and .numpy() both
construct a new view object that *points at the same memory*, which is why
they are effectively free (no data is copied) but also why mutating one
mutates the other. torch.tensor(array), in contrast, always allocates a new
buffer and copies the values in, which costs memory and time proportional to
the array's size but guarantees the two are afterwards fully independent.
The practical rule: use from_numpy()/.numpy() when handing data across the
boundary inside a single process and no aliasing bugs are wanted, and
torch.tensor() (or .clone()) whenever the two copies must be allowed to
diverge.
""")
