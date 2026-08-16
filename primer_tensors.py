"""Detour (not in the book) -- tensors explained as nested Java arrays."""
import torch

print("=" * 62)
print("A tensor is a nested array. .shape = length of each nesting level.")
print("=" * 62)

scalar = torch.tensor(7.0)
vector = torch.tensor([1., 2., 3.])
matrix = torch.tensor([[1., 2., 3.],
                       [4., 5., 6.]])
cube = torch.zeros(2, 3, 4)

for name, t, java in [
    ("scalar", scalar, "float x"),
    ("vector", vector, "float[3]"),
    ("matrix", matrix, "float[2][3]"),
    ("cube  ", cube,   "float[2][3][4]"),
]:
    print(f"{name}  shape={str(t.shape):<24}  java: {java}")

# ---------------------------------------------------------------- indexing
print("\nIndexing works exactly like Java, comma instead of ][ :")
print("  matrix[1][2] ->", matrix[1][2].item())
print("  matrix[1, 2] ->", matrix[1, 2].item(), " (same thing, PyTorch style)")

# ---------------------------------------------------------------- the shape we care about
print("\nThe shape this chapter builds toward:")
batch = torch.zeros(8, 4, 256)
print("  shape :", batch.shape)
print("  read  : 8 sentences  x  4 tokens each  x  256 numbers per token")
print("  java  : new float[8][4][256]")
print("  total :", batch.numel(), "individual numbers")

# ---------------------------------------------------------------- broadcasting
print("\nThe ONE non-Java trick you need: broadcasting.")
print("Adding a smaller tensor to a bigger one repeats it automatically.")
big = torch.zeros(3, 4)          # 3 rows of 4
small = torch.tensor([10., 20., 30., 40.])   # 1 row of 4
print("  big shape   :", big.shape)
print("  small shape :", small.shape)
print("  big + small :\n", big + small)
print("  -> 'small' got added to EVERY row. No loop written.")
print("  -> that is exactly how positional embeddings get applied in step 7.")
