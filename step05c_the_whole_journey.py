"""The whole journey, step by step: from one text file to one batch of shape [8, 4].

Run me:  python step05c_the_whole_journey.py

Every step prints what it produced, so the two numbers that cause all the
confusion stay separate and visible:
    1286  = how many windows the WHOLE FILE gives us   (set by the story length)
    8     = how many windows we GRAB at once            (set by batch_size, our knob)
"""
import tiktoken
import torch

# =====================================================================
# STEP 1 -- the file is just one long piece of English text
# =====================================================================
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("STEP 1  the file is plain text")
print(f"  characters in the file : {len(raw_text)}")
print(f"  it starts like this    : {raw_text[:60]!r}")

# =====================================================================
# STEP 2 -- turn the text into numbers (tokens)
# a model does math on numbers, not letters
# =====================================================================
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(raw_text)

print("\nSTEP 2  tokenize: text -> one flat list of numbers")
print(f"  how many numbers       : {len(token_ids)}")
print(f"  the first 8 numbers    : {token_ids[:8]}")
print(f"  decoded back to words  : {[tokenizer.decode([t]) for t in token_ids[:8]]}")

# =====================================================================
# STEP 3 -- chop that long list into small windows
#   max_length = how wide each window is        (fixed by the model = 4)
#   stride     = how far we jump for the next   (= 4, so no overlap)
# and STEP: each window also stores a "target" = the same window shifted by 1
# =====================================================================
max_length = 4
stride = 4

inputs_pile = []   # every window (the questions)
targets_pile = []  # every window shifted by 1 (the answer keys)

for i in range(0, len(token_ids) - max_length, stride):   # i = 0, 4, 8, ...
    input_chunk = token_ids[i:i + max_length]              # 4 numbers
    target_chunk = token_ids[i + 1:i + max_length + 1]     # same 4, shifted +1
    inputs_pile.append(torch.tensor(input_chunk))
    targets_pile.append(torch.tensor(target_chunk))

print("\nSTEP 3  slide a window (width 4, jump 4) over the whole list")
print(f"  windows created        : {len(inputs_pile)}   <- THIS is the 1286")
print(f"  each window is         : {max_length} tokens wide")

# =====================================================================
# STEP 4 -- decide how many windows to serve at once: batch_size
# this number is a knob YOU pick. It has nothing to do with the story.
# =====================================================================
batch_size = 8

print("\nSTEP 4  choose how many windows to grab at once")
print(f"  batch_size             : {batch_size}   <- THIS is the 8 (your knob)")
print(f"  so total batches       : {len(inputs_pile) // batch_size} "
      f"(1286 / 8, leftovers dropped)")

# =====================================================================
# STEP 5 -- grab ONLY the first batch: the first 8 windows out of 1286
# =====================================================================
first_batch_inputs = torch.stack(inputs_pile[:batch_size])    # windows 0..7
first_batch_targets = torch.stack(targets_pile[:batch_size])  # their answer keys

print("\nSTEP 5  take just the FIRST batch (windows 0..7 of the 1286)")
print(f"  inputs.shape           : {tuple(first_batch_inputs.shape)}  "
      f"= {batch_size} windows x {max_length} tokens")

print("\n  the 8 windows, decoded back into English:")
for r in range(batch_size):
    print(f"    row {r}: {tokenizer.decode(first_batch_inputs[r].tolist())!r}")

# =====================================================================
# THE TARGETS -- input vs its shifted partner, position by position
# =====================================================================
print("\nTHE TARGETS  each window's partner is the SAME window shifted by 1")
row0_in = first_batch_inputs[0].tolist()
row0_tg = first_batch_targets[0].tolist()
print(f"  input  (row 0): {[tokenizer.decode([t]) for t in row0_in]}")
print(f"  target (row 0): {[tokenizer.decode([t]) for t in row0_tg]}  <- slid left by 1")
print("\n  which means row 0 is really 4 practice questions:")
for pos in range(max_length):
    context = tokenizer.decode(row0_in[:pos + 1])
    answer = tokenizer.decode([row0_tg[pos]])
    print(f"    given {context!r:>32}  ->  predict {answer!r}")

print("\n" + "=" * 60)
print("RECAP:  1286 = windows the FILE gave us  (story length / 4)")
print("           8 = windows WE grabbed        (batch_size, our knob)")
print("        two different numbers, two different jobs.")
print("=" * 60)
