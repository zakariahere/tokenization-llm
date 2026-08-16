# Build an LLM from scratch — Chapter 2, worked slowly

My runnable notes for **Chapter 2** of Sebastian Raschka's *[Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)* — the input pipeline that turns raw text into the tensor an attention block consumes.

I'm a backend engineer (Java / Spring by day), so I learn this by relating it to things I already know: a tensor is a nested array, an embedding layer is a lookup table, a batch is `int[8][4]`. Every file is a small, **self-contained script that prints what it produced** — run it and watch the shapes appear.

> 📝 I write these up as I go — the code walkthrough lives in **[LLMs, the whole thing](https://blog.zakaria.lu/topics/llms)** and the intuition in **[Maths for LLMs](https://blog.zakaria.lu/topics/maths-for-llms)** on my blog.

## Run it

```bash
pip install torch tiktoken
python step07c_the_whole_whole_journey.py
```

Every script is standalone — start anywhere. If you only run one, run `step07c`: it's the whole chapter, end to end.

## The journey

| File | Book § | What it shows |
|---|---|---|
| `step01_tokenize.py` | 2.2 | Splitting raw text into tokens with a regex |
| `step02_vocab.py` | 2.3 | Building a vocabulary and a first tokenizer |
| `step03_special_tokens.py` | 2.4 | `<\|unk\|>` / `<\|endoftext\|>` for unknown words and document boundaries |
| `step04_bpe.py` | 2.5 | Byte-pair encoding — the tokenizer GPT-2 actually uses (`tiktoken`) |
| `step05_sliding_window.py` | 2.6 | Sliding window: one story → input/target training pairs |
| `step06_token_embeddings.py` | 2.7 | Token embeddings — an embedding layer is just a lookup table |
| `step07_positional.py` | 2.8 | Positional embeddings — giving the model a sense of word order |

### Detours & the "whole journey" scripts

These exist to kill specific confusions — the ones I actually got stuck on:

| File | Why it exists |
|---|---|
| `primer_tensors.py` | Tensors explained as nested Java arrays; `.shape` = the length of each nesting level |
| `step05b_what_is_the_8.py` | What the `8` and the `4` in shape `[8, 4]` actually *count* (two very different numbers) |
| `step05c_the_whole_journey.py` | Text → token IDs → windows → the batch `[8, 4]`, every step printed |
| `step07c_the_whole_whole_journey.py` | **The whole chapter, end to end:** text → `[8, 4]` IDs → `[8, 4, 256]` — meaning (token embeddings) *plus* order (positional embeddings) → the tensor attention consumes |

## The five numbers

Keeping these apart is the whole battle — they come from completely different places:

| Number | What it is | Where it comes from |
|---|---|---|
| `1286` | training windows | file length ÷ stride |
| `8` | windows per batch | `batch_size` — a knob you pick |
| `4` | tokens per window | `max_length` — the model's context length |
| `256` | numbers per token | embedding dim — room to carry meaning |
| `[8, 4, 256]` | the input tensor | all of the above, stacked |

## License & credits

- The code here is **adapted from** Sebastian Raschka's *[Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)*, which is licensed under the **Apache License 2.0** — rewritten and heavily commented for my own learning. See [`NOTICE`](NOTICE) for attribution and the changes made.
- This repository is released under the **Apache License 2.0** — see [`LICENSE`](LICENSE).
- The book's text and figures are **not** included and remain the copyright of the author and publisher.
- `the-verdict.txt` is *The Verdict* by Edith Wharton (1908) — public domain.
