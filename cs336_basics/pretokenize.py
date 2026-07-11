"""
One-off pre-tokenization: encode raw .txt files into token-ID arrays (.npy)
using the EXISTING vocab.pkl / merges.pkl, so training never re-tokenizes.

Usage (from cs336_basics/):
    python pretokenize.py                       # encodes train + valid with defaults
    python pretokenize.py --inputs foo.txt      # encode arbitrary file(s)

Output: for each input X.txt -> X.npy of dtype uint16 (vocab < 65536).
Load in training with:  np.load("TinyStoriesV2-GPT4-train.npy", mmap_mode="r")
"""
import argparse
import os
import time
import numpy as np
from tokenizer import Tokenizer

# vocab is 10000 (< 65536), so token IDs fit in uint16 -> ~half the RAM/disk of int32
TOKEN_DTYPE = np.uint16


def encode_file(tokenizer: Tokenizer, in_path: str, out_path: str) -> None:
    print(f"[{in_path}] encoding -> {out_path}")
    start = time.time()
    with open(in_path, "r", encoding="utf-8") as f:
        ids = np.fromiter(tokenizer.encode_iterable(f), dtype=TOKEN_DTYPE)

    max_id = int(ids.max()) if ids.size else 0
    if max_id > np.iinfo(TOKEN_DTYPE).max:
        raise ValueError(
            f"token id {max_id} exceeds {TOKEN_DTYPE.__name__} range; "
            f"widen TOKEN_DTYPE (e.g. np.int32)"
        )

    print("Saving to", out_path)
    np.save(out_path, ids)
    dur = time.time() - start
    print(
        f"[{in_path}] {ids.size:,} tokens  (max id {max_id})  "
        f"in {dur:.1f}s  ->  {out_path} "
        f"({os.path.getsize(out_path) / 1e6:.1f} MB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default="vocab.pkl")
    parser.add_argument("--merges", default="merges.pkl")
    parser.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[
            "TinyStories-train.txt",
            #"TinyStories-valid.txt",
        ],
        help="raw .txt files to encode; each X.txt -> X.npy",
    )
    args = parser.parse_args()

    # SAME tokenizer for every file -- train/val IDs must share one vocab
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special)

    for in_path in args.inputs:
        if not os.path.exists(in_path):
            print(f"[skip] {in_path} not found")
            continue
        out_path = os.path.splitext(in_path)[0] + ".npy"
        encode_file(tokenizer, in_path, out_path)

    print("done.")


if __name__ == "__main__":
    main()
