"""Step 1 (book section 2.2) -- splitting raw text into tokens."""
import re

# ---------------------------------------------------------------- 1a. tiny demo
text = "Hello, world. This, is a test."

# split on whitespace only -> punctuation stays glued to words
print("split on whitespace :", re.split(r'(\s)', text))

# also split on commas and periods -> punctuation becomes its own token
print("split on , and .    :", re.split(r'([,.]|\s)', text))

# drop the empty strings and the pure-whitespace bits
result = [t for t in re.split(r'([,.]|\s)', text) if t.strip()]
print("cleaned             :", result)

# ------------------------------------------------- 1b. the tokenizer we keep
# Handles: , . : ; ? _ ! " ( ) '  and the double-dash --
PATTERN = r'([,.:;?_!"()\']|--|\s)'


def tokenize(txt):
    pieces = re.split(PATTERN, txt)
    return [p.strip() for p in pieces if p.strip()]


print("\ntricky sample       :", tokenize("Hello, world. Is this-- a test?"))

# ------------------------------------------------- 1c. the real short story
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("\ncharacters in story :", len(raw_text))
print("first 99 characters :", raw_text[:99])

preprocessed = tokenize(raw_text)
print("\ntokens in story     :", len(preprocessed))
print("first 30 tokens     :", preprocessed[:30])
