import argparse
import io
from os import path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# hyper parameters
CONTEXT_WINDOW = 8
BATCH_SIZE = 32
N_MODEL = 32
EPOCH = 1
LEARNING_RATE = 1e-3


class Tokenizer:
    def __init__(self, chars: set):
        self.vocab_size = len(chars)
        self._i_to_s = {}
        self._s_to_i = {}
        for i, c in enumerate(sorted(list(chars))):
            self._i_to_s[i] = c
            self._s_to_i[c] = i

    def i_to_s(self, i: int) -> str:
        return self._i_to_s[i]

    def s_to_i(self, s: str) -> int:
        return self._s_to_i[s]

    def tokenize(self, text: str) -> torch.Tensor:
        res = torch.Tensor([self.s_to_i(s) for s in list(text)])
        return res.to(torch.long)


class ShakespearDataset(torch.utils.data.Dataset):
    def __init__(self, split: str):
        if split not in ["train", "dev", "test"]:
            raise "invalid split"

        with open(path.dirname(__file__) + f"/data/{split}.pt", "rb") as file:
            buffer = io.BytesIO(file.read())
            data = torch.load(buffer)
            self.x = data["x"]
            self.y = data["y"]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_MODEL)
        self.position_embedding = nn.Embedding(CONTEXT_WINDOW, N_MODEL)

        # this layer project the internal representation of dim N_MODEL
        # to a vocab_size dimension that represents the logits
        self.lm_head = nn.Linear(N_MODEL, vocab_size)

    # x is (B, T) we have to be careful that during inference, 1 <= T <= CONTEXT_WINDOW
    # we want to output logits (B, T, vocab_size)
    def forward(self, x):
        _, T = x.shape
        token_emb = self.token_embedding(x)  # (B, T, N_MODEL)
        position_emb = self.position_embedding(torch.arange(T))  # (T, N_MODEL)
        h = token_emb + position_emb  # (B, T, N_MODEL) thanks to broadcasting
        return self.lm_head(h)


def make_ds_tensors(tokenizer: Tokenizer, text: str, context_window=CONTEXT_WINDOW):
    tokenized_text = tokenizer.tokenize(text)
    x_list = []
    y_list = []

    for i in tqdm(range(len(tokenized_text) - context_window - 1)):
        input = tokenized_text[i : i + context_window]
        target = tokenized_text[i + 1 : i + context_window + 1]

        x_list.append(input)
        y_list.append(target)

    return (torch.stack(x_list, 0), torch.stack(y_list, 0))


def make_dataset(tokenizer, text):
    train_text = text[: int(len(text) * 0.8)]
    dev_text = text[int(len(text) * 0.8) : int(len(text) * 0.9)]
    test_text = text[int(len(text) * 0.9) :]

    x_train, y_train = make_ds_tensors(tokenizer, train_text)
    torch.save({"x": x_train, "y": y_train}, "data/train.pt")
    print("dev done")

    x_dev, y_dev = make_ds_tensors(tokenizer, dev_text)
    torch.save({"x": x_dev, "y": y_dev}, "data/dev.pt")
    print("valid done")

    x_test, y_test = make_ds_tensors(tokenizer, test_text)
    torch.save({"x": x_test, "y": y_test}, "data/test.pt")
    print("test done")


def get_loss(logits, Y):
    # we need to reshape Y and logits to make it fit to cross_entropy_loss
    # logits (B, T, C) -> (B*T, C)
    # Y (B, T) -> (B*T,)
    B, T, C = logits.shape
    return F.cross_entropy(logits.view((B * T, C)), Y.view((B * T,)))


@torch.no_grad()
def eval(model: nn.Module, dev_dl):
    model.eval()
    losses = []
    for X, Y in dev_dl:
        logits = model(X)
        loss = get_loss(logits, Y)
        losses.append(loss.item())

    print(f"dev loss={torch.tensor(losses).mean().item()}")
    model.train()


def train(model: nn.Module, train_dl, dev_dl):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for e in range(EPOCH):
        print(f"--- EPOCH {e + 1} ---")
        for X, Y in train_dl:
            # X is (B, T)
            # Y is (B, T)
            logits = model(X)  # logits is (B, T, vocab_size C)
            loss = get_loss(logits, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    eval(model, dev_dl)


def run_data(tokenizer, text):
    make_dataset(tokenizer, text)


def run_train(tokenizer):
    train_dl = torch.utils.data.DataLoader(
        ShakespearDataset("train"), batch_size=BATCH_SIZE, shuffle=True
    )
    dev_dl = torch.utils.data.DataLoader(
        ShakespearDataset("dev"), batch_size=BATCH_SIZE
    )
    model = Model(tokenizer.vocab_size)
    train(model, train_dl, dev_dl)
    torch.save(model, "data/model.pt")


@torch.no_grad()
def run_infer(tokenizer: Tokenizer):
    with open(path.dirname(__file__) + "/data/model.pt", "rb") as file:
        buffer = io.BytesIO(file.read())
        model = torch.load(buffer)

    model.eval()
    res = []
    context = torch.zeros((1, 1), dtype=torch.long)
    for _ in range(1000):
        context = context[:, -CONTEXT_WINDOW:]
        logits = model(context)  # logits is (1, T, VOCAB_SIZE)
        probs = F.softmax(logits, dim=2)  # probs is (1, T, VOCAB_SIZE)
        next_idx = torch.multinomial(
            probs[:, -1, :], 1
        )  # draw 1 sample from the last T next_idx is (1, 1)
        res.append(next_idx.item())
        context = torch.cat([context, next_idx], dim=1)

    print("".join(tokenizer.i_to_s(i) for i in res))


def run(args):
    t = None
    text = None
    try:
        with open(path.dirname(__file__) + "/data/input.txt") as f:
            text = f.read()
            # TODO persist tokenizer state to file so we don't need the raw
            # input at training or inference
            t = Tokenizer(set(text))
    except FileNotFoundError:
        print("raw dataset is not found. Please see data/README for more info")
        sys.exit(1)

    if args.mode == "data":
        run_data(t, text)

    if args.mode == "train":
        run_train(t)

    if args.mode == "infer":
        run_infer(t)


def main():
    parser = argparse.ArgumentParser(
        prog="tiny-shakespear",
        description="generate shakespear like text using a transformer architecture",
    )

    parser.add_argument(
        "-m",
        "--mode",
        nargs="?",
        choices=["data", "train", "infer"],
        default="data",
        help="selects which mode to run in. default is data",
    )

    run(parser.parse_args())


if __name__ == "__main__":
    main()
