"""Step 4 (book section 2.5) -- byte pair encoding, the tokenizer GPT-2 actually uses."""
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
print("vocabulary size :", tokenizer.n_vocab)

# ------------------------------- 4a. same sentence as step 3, no crash, no <|unk|>
text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace.")
ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("\nencoded :", ids)
print("decoded :", tokenizer.decode(ids))

# ------------------------------- 4b. see WHERE it split each word
print("\n--- how a common sentence splits ---")
for i in tokenizer.encode("Hello, do you like tea?"):
    print(f"  {i:>6}  ->  {tokenizer.decode([i])!r}")

# ------------------------------- 4c. exercise 2.1: pure gibberish
print("\n--- exercise 2.1: 'Akwirw ier' (not a word in any language) ---")
gibberish = "Akwirw ier"
ids = tokenizer.encode(gibberish)
print("token ids :", ids)
for i in ids:
    print(f"  {i:>6}  ->  {tokenizer.decode([i])!r}")
print("rebuilt   :", repr(tokenizer.decode(ids)))

# ------------------------------- 4d. one word, three different token counts
print("\n--- common vs rare words ---")
for w in [" the", " painted", " pardonable", " antidisestablishmentarianism"]:
    ids = tokenizer.encode(w)
    parts = [tokenizer.decode([i]) for i in ids]
    print(f"  {w!r:>32}  ->  {len(ids)} token(s)  {parts}")
