import argparse
import io
from os import path
import sys

import torch
from tqdm import tqdm

CONTEXT_WINDOW = 8
BATCH_SIZE = 32


class Tokenizer:
    def __init__(self, chars: set):
        self.vocab_size = len(chars)
        self._i_to_s = {}
        self._s_to_i = {}
        for i, c in enumerate(sorted(list(chars))):
            self._i_to_s[i + 1] = c
            self._s_to_i[c] = i + 1

    def i_to_s(self, i: int) -> str:
        return self._i_to_s[i]

    def s_to_i(self, s: str) -> int:
        return self._s_to_i[s]

    def tokenize(self, text: str) -> torch.Tensor:
        res = torch.Tensor([self.s_to_i(s) for s in list(text)])
        return res.to(torch.int8)


class ShakespearDataset(torch.utils.data.Dataset):
    def __init__(self, split: str):
        if split not in ["dev", "test", "valid"]:
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
    dev_text = text[: int(len(text) * 0.8)]
    valid_text = text[int(len(text) * 0.8) : int(len(text) * 0.9)]
    test_text = text[int(len(text) * 0.9) :]

    x_dev, y_dev = make_ds_tensors(tokenizer, dev_text)
    torch.save({"x": x_dev, "y": y_dev}, "data/dev.pt")
    print("dev done")

    x_valid, y_valid = make_ds_tensors(tokenizer, valid_text)
    torch.save({"x": x_valid, "y": y_valid}, "data/valid.pt")
    print("valid done")

    x_test, y_test = make_ds_tensors(tokenizer, test_text)
    torch.save({"x": x_test, "y": y_test}, "data/test.pt")
    print("test done")


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
        make_dataset(t, text)
        sys.exit(0)

    dev_ds = ShakespearDataset("dev")
    dev_dl = torch.utils.data.DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=True)
    for X, Y in dev_dl:
        for i in range(BATCH_SIZE):
            x, y = X[i], Y[i]
            print(x, y)
            for j in range(y.shape[0]):
                print(
                    "".join([t.i_to_s(i) for i in x[: j + 1].tolist()]),
                    "->",
                    t.i_to_s(y[j].item()),
                )
            break
        break


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
