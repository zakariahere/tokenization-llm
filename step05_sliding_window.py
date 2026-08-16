"""Step 5 (book section 2.6) -- sliding window: turning one story into training pairs."""
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """max_length = how many tokens the model sees at once.
       stride     = how far the window jumps for the next example."""

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):                    # how many rows exist
        return len(self.input_ids)

    def __getitem__(self, idx):           # fetch row number idx
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers)


# everything below only runs when you execute THIS file directly,
# so other steps can `from step05_sliding_window import create_dataloader_v1`
# without triggering all these printouts.
if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    enc_text = tokenizer.encode(raw_text)
    print("BPE tokens in the whole story :", len(enc_text))

    # ------------------------- 5a. input / target = the same list, shifted by 1
    enc_sample = enc_text[50:]          # skip 50 tokens, nicer passage to look at
    context_size = 4

    x = enc_sample[:context_size]
    y = enc_sample[1:context_size + 1]
    print("\nx (input)  :", x)
    print("y (target) :", y, "   <- x shifted left by one")

    print("\nthe 4 prediction tasks hiding in that single row:")
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(f"  {tokenizer.decode(context)!r:>40}  ---->  "
              f"{tokenizer.decode([desired])!r}")

    # ------------------------- 5c. stride = 1 : windows overlap heavily
    print("\n" + "=" * 62)
    print("batch_size=1, max_length=4, stride=1")
    print("=" * 62)
    loader = create_dataloader_v1(raw_text, batch_size=1, max_length=4,
                                  stride=1, shuffle=False)
    it = iter(loader)
    inputs, targets = next(it)
    print("batch 1 input :", inputs.tolist(), " target:", targets.tolist())
    inputs, targets = next(it)
    print("batch 2 input :", inputs.tolist(), " target:", targets.tolist())
    print("-> batch 2 shifted by exactly 1. Windows overlap almost completely.")

    # ------------------------- 5d. stride = max_length : no overlap at all
    print("\n" + "=" * 62)
    print("batch_size=8, max_length=4, stride=4   <- the useful setting")
    print("=" * 62)
    loader = create_dataloader_v1(raw_text, batch_size=8, max_length=4,
                                  stride=4, shuffle=False)
    inputs, targets = next(iter(loader))
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)
    print("\ninputs.shape :", inputs.shape,
          "  = 8 sentences x 4 tokens  (java: int[8][4])")
    print("rows in the whole dataset :", len(loader.dataset))
