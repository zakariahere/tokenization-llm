"""Step 6 (book section 2.7) -- an embedding layer is just a lookup table."""
import torch

# ------------------------------- 6a. tiny toy: 6 words, 3 numbers each
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print("the whole lookup table (java: float[6][3]):")
print(embedding_layer.weight)
print("\nshape:", embedding_layer.weight.shape,
      " -> one ROW per vocabulary word, one COLUMN per dimension")

# ------------------------------- 6b. looking up one token
print("\nlook up token id 3:")
print(embedding_layer(torch.tensor([3])))
print("-> that is literally row 3 of the table above. No math, just a row fetch.")

# ------------------------------- 6c. looking up four tokens at once
input_ids = torch.tensor([2, 3, 5, 1])
print("\nlook up ids [2, 3, 5, 1] together:")
print(embedding_layer(input_ids))
print("shape:", embedding_layer(input_ids).shape, " -> 4 tokens x 3 numbers")

# ------------------------------- 6d. the real thing
print("\n" + "=" * 66)
print("now for real: GPT-2 vocabulary, 256 dimensions")
print("=" * 66)
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print("table shape :", token_embedding_layer.weight.shape)
print("parameters  :", f"{token_embedding_layer.weight.numel():,} numbers to learn")

import tiktoken
from step05_sliding_window import create_dataloader_v1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

max_length = 4
loader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length,
                              stride=max_length, shuffle=False)
inputs, targets = next(iter(loader))

print("\ntoken ids in  :", inputs.shape, "  java: int[8][4]")
token_embeddings = token_embedding_layer(inputs)
print("embeddings out:", token_embeddings.shape, "  java: float[8][4][256]")
print("\n-> the 256 is 'how many numbers describe the meaning of one token'")
print("-> GPT-2 small uses 768. GPT-3 uses 12,288. We use 256 to keep it light.")

# ------------------------------- 6e. same token id -> same vector, always
print("\nnote: the SAME id always returns the SAME row, wherever it appears:")
a = token_embedding_layer(torch.tensor([345]))
b = token_embedding_layer(torch.tensor([345]))
print("  identical? ", torch.equal(a, b), " <- this is the problem step 7 fixes")
