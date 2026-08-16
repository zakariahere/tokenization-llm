"""Step 7 (book section 2.8) -- give the model a sense of word order."""
import torch
from step05_sliding_window import create_dataloader_v1

vocab_size = 50257
output_dim = 256
max_length = 4

torch.manual_seed(123)
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# ------------------------------- 7a. a SECOND table, indexed by slot number
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("torch.arange(4) =", torch.arange(context_length).tolist(),
      "  <- 'slot 0, slot 1, slot 2, slot 3'")
print("pos_embeddings shape :", pos_embeddings.shape, " -> 4 slots x 256 numbers")

# ------------------------------- 7b. the proof: same word, different slots
print("\n" + "=" * 66)
print("PROOF -- feed the SAME token id four times in a row")
print("=" * 66)
same_word = torch.tensor([[588, 588, 588, 588]])          # " like" x4
tok = token_embedding_layer(same_word)
print("token embeddings, first 4 numbers of each slot:")
for s in range(4):
    print(f"  slot {s}: {[round(v, 4) for v in tok[0, s, :4].tolist()]}")
print("  -> all four rows IDENTICAL. word order is invisible.")

final = tok + pos_embeddings
print("\nafter adding positional embeddings:")
for s in range(4):
    print(f"  slot {s}: {[round(v, 4) for v in final[0, s, :4].tolist()]}")
print("  -> all four rows now DIFFERENT. the model can tell slots apart.")

# ------------------------------- 7c. the real pipeline, end to end
print("\n" + "=" * 66)
print("the full chapter 2 pipeline, one batch")
print("=" * 66)
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

loader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length,
                              stride=max_length, shuffle=False)
inputs, targets = next(iter(loader))

token_embeddings = token_embedding_layer(inputs)

print("  raw text            -> a python str")
print("  token ids           ->", tuple(inputs.shape), "         int[8][4]")
print("  token embeddings    ->", tuple(token_embeddings.shape), "   float[8][4][256]")
print("  positional embeds   ->", tuple(pos_embeddings.shape), "      float[4][256]")

input_embeddings = token_embeddings + pos_embeddings      # broadcasting!
print("  input embeddings    ->", tuple(input_embeddings.shape), "   float[8][4][256]")

print("\nbroadcasting note: [4][256] was added to ALL 8 rows automatically.")
print("every sentence in the batch shares the same 4 position vectors,")
print("because slot 0 means 'slot 0' no matter which sentence you are in.")
print("\nthis tensor is exactly what chapter 3 feeds into attention.")
