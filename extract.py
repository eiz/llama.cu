#!/usr/bin/env python3
import fire
import json
import os
import struct
import torch
from sentencepiece import SentencePieceProcessor


def extract_weights(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("loading weights...")
    state_dict = torch.load(input_file, map_location="cpu")

    for key, tensor in state_dict.items():
        tensor = tensor.to(dtype=torch.float16)
        file_name = f"{key}__{'_'.join(map(str, tensor.shape))}"
        file_path = os.path.join(output_dir, file_name)

        print(f"writing {file_path}...")
        with open(file_path, "wb") as f:
            f.write(tensor.numpy().tobytes())


def extract_params(input_file, output_file):
    with open(input_file, "r") as f_in, open(output_file, "wb") as f_out:
        config = json.load(f_in)
        f_out.write(
            struct.pack(
                "iiiiif",
                config["dim"],
                config["multiple_of"],
                config["n_heads"],
                config["n_layers"],
                32000,
                config["norm_eps"],
            )
        )


def extract_vocab(input_file, output_file):
    tokenizer = SentencePieceProcessor(input_file)
    with open(output_file, "wb") as f:
        pieces = [
            tokenizer.id_to_piece(i).replace("\u2581", " ").encode("utf-8")
            for i in range(tokenizer.vocab_size())
        ]
        cum_offset = 4 + 8 * tokenizer.vocab_size()
        i = 0
        f.write(struct.pack("i", tokenizer.vocab_size()))
        for piece in pieces:
            f.write(struct.pack("if", cum_offset, tokenizer.get_score(i)))
            i += 1
            cum_offset += len(piece) + 1
        for piece in pieces:
            f.write(piece)
            f.write(b"\x00")


if __name__ == "__main__":
    fire.Fire(
        {
            "weights": extract_weights,
            "params": extract_params,
            "vocab": extract_vocab,
        }
    )
