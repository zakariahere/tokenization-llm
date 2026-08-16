"""Step 3 (book section 2.4) -- special tokens for unknown words and doc boundaries."""
import re

PATTERN = r'([,.:;?_!"()\']|--|\s)'

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = [p.strip() for p in re.split(PATTERN, raw_text) if p.strip()]

# ------------------------------- 3a. vocabulary + 2 extra "words" at the end
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: i for i, token in enumerate(all_tokens)}

print("vocab size now :", len(vocab), "(was 1130)")
print("last 5 entries :", list(vocab.items())[-5:])


# ------------------------------- 3b. tokenizer that never crashes
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        pieces = [p.strip() for p in re.split(PATTERN, text) if p.strip()]
        # the one new line vs V1: swap anything unknown for <|unk|>
        pieces = [p if p in self.str_to_int else "<|unk|>" for p in pieces]
        return [self.str_to_int[p] for p in pieces]

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        return re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)


tokenizer = SimpleTokenizerV2(vocab)

# ------------------------------- 3c. glue two unrelated sentences together
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

print("\ninput   :", text)
ids = tokenizer.encode(text)
print("encoded :", ids)
print("decoded :", tokenizer.decode(ids))
print("\n1130 = <|endoftext|>, 1131 = <|unk|>")
print("-> the story never uses the words 'Hello' or 'palace', so both became <|unk|>")
