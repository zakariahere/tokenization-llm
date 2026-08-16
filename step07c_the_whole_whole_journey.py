"""The WHOLE WHOLE journey, end to end: from one text file to the [8, 4, 256]
tensor that the attention mechanism consumes -- all of chapter 2 in one run.

Run me:  python step07c_the_whole_whole_journey.py

step05c stopped at the batch of token IDs, shape [8, 4]. This file picks up
exactly there and FINISHES chapter 2: it turns those bare IDs into MEANING
(token embeddings, section 2.7) and then stamps them with ORDER (positional
embeddings, section 2.8).

Five numbers cause all the confusion. They come from DIFFERENT places, so we
keep them visually separate and let each step print its own shape:

    1286  windows the FILE gives us      (story length / stride)
    8     windows we GRAB per batch       (batch_size -- a knob YOU pick)
    4     tokens per window               (max_length -- the model's context)
    256   numbers describing one token    (embedding dim -- room for meaning)
    ->    final input tensor: [8, 4, 256]
"""
import tiktoken
import torch

# Seed once so the random lookup tables (and every number printed below) are
# identical on every run. We create the TOKEN table before the POSITION table,
# the same order as step07_positional.py, so the proof numbers match.
torch.manual_seed(123)

# =====================================================================
# STEP 1 -- the file is just one long piece of English text
# =====================================================================
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("STEP 1  the file is plain text")
print(f"  characters in the file : {len(raw_text)}")

# =====================================================================
# STEP 2 -- turn the text into numbers (token IDs)
# A network does math on numbers, not letters. Each ID is just a ROW NUMBER
# in the tokenizer's dictionary -- NOT a quantity. id 8887 (" tea") is not
# "bigger" than id 345 (" you"); it's simply a different shelf.
# =====================================================================
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(raw_text)

print("\nSTEP 2  tokenize: text -> one flat list of numbers")
print(f"  how many numbers       : {len(token_ids)}")
print(f"  first 8 numbers        : {token_ids[:8]}")

# =====================================================================
# STEP 3 -- slide a window over the list to make training examples
#   max_length = how wide each window is    (= 4 tokens)
#   stride     = how far we jump each time  (= 4, so windows don't overlap)
# Each window also stores a "target": the SAME window shifted right by one,
# because the model's job is "given these tokens, predict the next one".
# =====================================================================
max_length = 4
stride = 4

inputs_pile = []    # every window            (the questions)
targets_pile = []   # every window shifted +1 (the answer keys)

for i in range(0, len(token_ids) - max_length, stride):   # i = 0, 4, 8, ...
    input_chunk = token_ids[i:i + max_length]             # 4 numbers
    target_chunk = token_ids[i + 1:i + max_length + 1]    # same 4, shifted +1
    inputs_pile.append(torch.tensor(input_chunk))
    targets_pile.append(torch.tensor(target_chunk))

print("\nSTEP 3  sliding window -> many small training examples")
print(f"  windows created        : {len(inputs_pile)}   <- the 1286 (set by the FILE)")
print(f"  each window is         : {max_length} tokens wide")

# =====================================================================
# STEP 4 -- batch_size: how many windows we serve at once.
# This is YOUR knob. It has nothing to do with the story's length.
# =====================================================================
batch_size = 8
print("\nSTEP 4  batch_size = how many windows we grab at once")
print(f"  batch_size             : {batch_size}   <- the 8 (YOUR knob)")
print(f"  total batches          : {len(inputs_pile) // batch_size}")

# =====================================================================
# STEP 5 -- grab the FIRST batch: 8 windows of 4 token IDs -> shape [8, 4]
# =====================================================================
inputs = torch.stack(inputs_pile[:batch_size])     # [8, 4] token IDs
targets = torch.stack(targets_pile[:batch_size])   # [8, 4] answer keys

print("\nSTEP 5  the first batch of token IDs")
print(f"  inputs.shape           : {tuple(inputs.shape)}  = {batch_size} windows x {max_length} tokens")
print(f"  row 0 decoded          : {tokenizer.decode(inputs[0].tolist())!r}")

# The target is the same row shifted by one, so ONE window is really 4
# next-token questions stacked together.
row0_in, row0_tg = inputs[0].tolist(), targets[0].tolist()
print("  row 0 hides 4 questions:")
for pos in range(max_length):
    ctx = tokenizer.decode(row0_in[:pos + 1])
    ans = tokenizer.decode([row0_tg[pos]])
    print(f"    given {ctx!r:>26} -> predict {ans!r}")

# =====================================================================
# STEP 6 -- token embeddings (section 2.7): turn each ID into a VECTOR
# An embedding layer is just a lookup table: a float[vocab_size][256] grid.
# The token ID is the ROW NUMBER; the row it fetches is that token's
# meaning-vector. No math, just a row fetch. (Those numbers are random noise
# right now -- they only become meaningful during training, chapter 5.)
# =====================================================================
vocab_size = 50257     # GPT-2's vocabulary
output_dim = 256       # how many numbers describe one token's meaning

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)   # [8, 4] -> [8, 4, 256]

print("\nSTEP 6  token embeddings: look up each ID -> a 256-number row")
print(f"  table shape            : {tuple(token_embedding_layer.weight.shape)}"
      f"  ({token_embedding_layer.weight.numel():,} learnable numbers)")
print(f"  IDs in                 : {tuple(inputs.shape)}")
print(f"  embeddings out         : {tuple(token_embeddings.shape)}   <- a 256-vector under every token")

# =====================================================================
# STEP 7 -- positional embeddings (section 2.8): stamp each token with WHERE
# it sits. Build a SECOND lookup table, indexed by SLOT (0,1,2,3) instead of
# by token ID -- one row per position in the window.
#
# Mental model: chairs and stickers. There are 4 seats; each seat has a fixed
# "sticker" (a 256-vector). Whoever sits there gets that sticker ADDED on top.
# The sticker is glued to the SEAT, not to the word -- so the same word in two
# seats ends up different.
#
# We never add the two TABLES (they're different sizes: 50257x256 vs 4x256).
# We add the rows we looked up -- and broadcasting adds the same 4 seat-vectors
# to ALL 8 windows, because "slot 0" means "slot 0" in every window.
# =====================================================================
context_length = max_length    # 4 seats, because a window is 4 tokens long
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)   # [4, 256]
pos_embeddings = pos_embedding_layer(torch.arange(context_length))     # slots 0..3 -> [4, 256]

input_embeddings = token_embeddings + pos_embeddings   # [8,4,256] + [4,256] -> [8,4,256]

print("\nSTEP 7  positional embeddings: a 2nd table, indexed by SLOT, added on")
print(f"  position table shape   : {tuple(pos_embedding_layer.weight.shape)}   (one row per seat)")
print(f"  token   {tuple(token_embeddings.shape)}")
print(f"  + pos   {tuple(pos_embeddings.shape)}      (broadcast across all {batch_size} windows)")
print(f"  = input {tuple(input_embeddings.shape)}   <- what attention consumes next")

# =====================================================================
# WHY POSITION MATTERS -- proof it makes identical tokens distinct.
# Feed the SAME token id into all 4 slots. With only token embeddings the 4
# rows are identical (order is invisible). After adding the slot-vectors they
# differ -- purely because each seat contributed a different sticker.
# =====================================================================
print("\nPROOF  same token in 4 slots: identical until position is added")
same = torch.tensor([[588, 588, 588, 588]])   # " like" x4, a single window
tok = token_embedding_layer(same)             # [1, 4, 256]
fixed = tok + pos_embeddings                  # add the 4 seat-vectors
print("  token embeddings (first 3 numbers of each slot):")
for s in range(max_length):
    print(f"    slot {s}: {[round(v, 3) for v in tok[0, s, :3].tolist()]}   <- all identical")
print("  after + positional embeddings:")
for s in range(max_length):
    print(f"    slot {s}: {[round(v, 3) for v in fixed[0, s, :3].tolist()]}   <- now all different")

# =====================================================================
# RECAP -- the whole of chapter 2 on one screen
# =====================================================================
print("\n" + "=" * 64)
print("THE WHOLE WHOLE JOURNEY")
print(f"  text        the-verdict.txt        a python str")
print(f"  token IDs   {len(token_ids):<5} numbers          one flat list")
print(f"  windows     {len(inputs_pile):<5}                  file / stride")
print(f"  batch       [{batch_size}, {max_length}]                 batch_size x window")
print(f"  + meaning   [{batch_size}, {max_length}, {output_dim}]            token embeddings  (WHAT)")
print(f"  + order     [{batch_size}, {max_length}, {output_dim}]            positional embeds (WHERE)")
print("  " + "-" * 60)
print(f"  RESULT      input_embeddings {tuple(input_embeddings.shape)}  ->  attention (ch.3)")
print("=" * 64)
