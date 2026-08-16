"""Step 2 (book section 2.3) -- building a vocabulary and a first tokenizer."""
import re

PATTERN = r'([,.:;?_!"()\']|--|\s)'

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = [p.strip() for p in re.split(PATTERN, raw_text) if p.strip()]

# ------------------------------------------------- 2a. build the vocabulary
all_words = sorted(set(preprocessed))          # unique + alphabetical
vocab = {token: i for i, token in enumerate(all_words)}

print("total tokens in story :", len(preprocessed))
print("UNIQUE tokens (vocab) :", len(vocab))
print("\nfirst 6 vocab entries :", list(vocab.items())[:6])
print("entries 48-51         :", list(vocab.items())[48:51])


# ------------------------------------------------- 2b. the tokenizer class
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab                                  # word  -> id
        self.int_to_str = {i: s for s, i in vocab.items()}        # id    -> word

    def encode(self, text):
        pieces = [p.strip() for p in re.split(PATTERN, text) if p.strip()]
        return [self.str_to_int[s] for s in pieces]

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # " hello ." -> " hello."   (glue punctuation back onto the word)
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)


tokenizer = SimpleTokenizerV1(vocab)

sample = '"It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
ids = tokenizer.encode(sample)
print("\noriginal :", sample)
print("encoded  :", ids)
print("decoded  :", tokenizer.decode(ids))

# ------------------------------------------------- 2c. where it breaks
print("\n--- now a word the story never uses ---")
try:
    tokenizer.encode("Hello, do you like tea?")
except KeyError as e:
    print("KeyError:", e, "  <- not in the vocabulary, tokenizer dies")
