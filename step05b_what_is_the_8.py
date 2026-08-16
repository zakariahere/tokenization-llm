"""Detour -- what do the 8 and the 4 in shape [8, 4] actually count?"""
import tiktoken
import torch
from step05_sliding_window import create_dataloader_v1

tokenizer = tiktoken.get_encoding("gpt2")
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# ---------------------------------------------- A. decode the batch back to text
loader = create_dataloader_v1(raw_text, batch_size=8, max_length=4,
                              stride=4, shuffle=False)
inputs, targets = next(iter(loader))
print("shape:", inputs.shape, "\n")
print("the 8 ROWS, decoded back into English:")
for r in range(inputs.shape[0]):
    print(f"  row {r}: {tokenizer.decode(inputs[r].tolist())!r}")

print("\nglued together they ARE the start of the story:")
print(" ", repr(tokenizer.decode(inputs.flatten().tolist())))
print("  (8 rows x 4 tokens = 32 tokens = a subset, exactly as you guessed)")

# ---------------------------------------------- B. but the rows are INDEPENDENT
print("\n" + "=" * 66)
print("now the same thing with shuffle=True -- rows come from all over the book")
print("=" * 66)
torch.manual_seed(123)
loader = create_dataloader_v1(raw_text, batch_size=8, max_length=4,
                              stride=4, shuffle=True)
inputs, _ = next(iter(loader))
for r in range(inputs.shape[0]):
    print(f"  row {r}: {tokenizer.decode(inputs[r].tolist())!r}")
print("\n-> the 8 rows have NOTHING to do with each other.")
print("-> they are 8 unrelated homework questions answered side by side.")

# ---------------------------------------------- C. the 8 is a free choice
print("\n" + "=" * 66)
print("the 4 is fixed by the model. the 8 is just a knob you turn:")
print("=" * 66)
for bs in (1, 2, 8, 32):
    ld = create_dataloader_v1(raw_text, batch_size=bs, max_length=4,
                              stride=4, shuffle=False)
    xb, _ = next(iter(ld))
    print(f"  batch_size={bs:<3} -> shape {str(tuple(xb.shape)):<10} "
          f"{len(ld)} batches to read the whole story once")
