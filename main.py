import argparse
from os import path
import sys

import torch

NO_CHAR = "#"
CONTEXT_WINDOW = 8


class Tokenizer:
    def __init__(self, chars: set):
        self.vocab_size = len(chars) + 1
        self._i_to_s = {0: NO_CHAR}
        self._s_to_i = {NO_CHAR: 0}
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


# class ShakespearDataset(torch.utils.data.Dataset):
#     def __init__(self, tokenizer: Tokenizer, text: str, context_window=CONTEXT_WINDOW):
#         tokenized_text = tokenizer.tokenize(text)
#         x_list = []
#         y_list = []
#         mask = (
#             torch.ones(context_window, context_window)
#             .tril()
#             .flip(1)
#             .to(dtype=torch.int64)
#         )
#         for i in range(len(tokenized_text) - context_window - 1):
#             if i % 100 == 0:
#                 print(i, len(tokenized_text) - context_window - 1)

#             input = tokenized_text[i : i + context_window]
#             target = tokenized_text[i + context_window]

#             x = input.repeat(context_window, 1) * mask
#             y = target.repeat(context_window)
#             x_list.append(x)
#             y_list.append(y)

#         self.X = torch.cat(x_list)
#         self.Y = torch.cat(y_list)

#     def __len__(self):
#         return self.X.shape[0]

#     def __getitem__(self, index):
#         return self.X[index], self.Y[index]


def make_ds_tensors(tokenizer: Tokenizer, text: str, context_window=CONTEXT_WINDOW):
    tokenized_text = tokenizer.tokenize(text)
    x_list = []
    y_list = []
    mask = (
        torch.ones(context_window, context_window).tril().flip(1).to(dtype=torch.int8)
    )
    for i in range(len(tokenized_text) - context_window - 1):
        if i % 10000 == 0:
            print(i, len(tokenized_text) - context_window - 1)

        input = tokenized_text[i : i + context_window]
        target = tokenized_text[i + context_window]

        x = input.repeat(context_window, 1) * mask
        y = target.repeat(context_window)
        x_list.append(x)
        y_list.append(y)

    return (torch.cat(x_list), torch.cat(y_list))


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
    if args.mode != "data":
        print("not implemented")
        sys.exit(1)
        # raise Exception("not implemented")

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


def main():
    parser = argparse.ArgumentParser(
        prog="tiny-shakespear",
        description="generate shakespear like text using a transformer architecture",
    )

    # parser.add_argument("-h", "--help", action="help")
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
