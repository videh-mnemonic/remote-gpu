"""Bounded compatibility gate for modded-nanogpt's runtime kernel module."""

import argparse

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()
torch.cuda.set_device(args.device)

import triton_kernels  # noqa: E402,F401

print("MODDED_KERNEL_IMPORT_OK")
