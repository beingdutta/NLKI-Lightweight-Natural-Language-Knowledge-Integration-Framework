#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train ColBERT with CLI args.

Example:
python ColBERT-Finetune.py \
  --queries Contrastive-Training-Data/testData160000Final/queries_train_FAISS_1pos_1neg.tsv \
  --collection Contrastive-Training-Data/testData160000Final/collection_train_FAISS_1pos_1neg.tsv \
  --mapping Contrastive-Training-Data/testData160000Final/mapping_train_FAISS_1pos_1neg.jsonl \
  --checkpoint colbert-ir/colbertv1.9 \
  --device 0 \
  --bsize 32 --lr 1e-4 --warmup 20000 --doc-maxlen 180 --dim 128 \
  --nway 2 --accumsteps 1 --similarity cosine --use-ib-negatives \
  --inspect
"""

import argparse
import os
import sys
import json
import pandas as pd
import torch

# third-party deps used by original script
import faiss  # noqa: F401  # imported to ensure availability
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer


def positive_file(path: str) -> str:
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ColBERT on custom triples/queries/collection")
    # data
    p.add_argument("--queries", type=positive_file, required=True, help="Path to queries .tsv")
    p.add_argument("--collection", type=positive_file, required=True, help="Path to collection .tsv")
    p.add_argument("--mapping", type=positive_file, required=True, help="Path to mapping .jsonl (triples)")
    # training / model
    p.add_argument("--checkpoint", type=str, default="colbert-ir/colbertv1.9",
                   help="Model checkpoint to start from (e.g., 'bert-base-uncased' or a ColBERT ckpt)")
    p.add_argument("--bsize", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=20_000)
    p.add_argument("--doc-maxlen", type=int, default=180)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--nway", type=int, default=2, help="#ways for in-batch negatives")
    p.add_argument("--accumsteps", type=int, default=1)
    p.add_argument("--similarity", type=str, default="cosine", choices=["cosine", "l2", "dot"])
    p.add_argument("--attend-to-mask-tokens", action="store_true",
                   help="If set, attend to [MASK] tokens (default: False)")
    p.add_argument("--use-ib-negatives", action="store_true",
                   help="Enable in-batch negatives")
    # infra
    p.add_argument("--nranks", type=int, default=1, help="DDP world size (use 1 for single GPU)")
    p.add_argument("--device", type=int, default=0, help="CUDA device index; use -1 for CPU")
    # utils
    p.add_argument("--inspect", action="store_true",
                   help="Print basic stats of queries/collection/mapping before training")
    return p.parse_args()


def inspect_data(queries_path: str, collection_path: str, mapping_path: str) -> None:
    print("Inspecting datasets...")
    qdf = pd.read_csv(queries_path, sep="\t")
    cdf = pd.read_csv(collection_path, sep="\t")
    print(f"Queries:   {len(qdf)} rows")
    print(f"Collection:{len(cdf)} rows")

    # Count lines in mapping and preview a couple
    n_lines = 0
    preview = 5
    print("Mapping preview:")
    with open(mapping_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i <= preview:
                print(line.strip())
            n_lines += 1
    print(f"Mapping:   {n_lines} lines\n")


def main():
    args = parse_args()

    # Device
    if args.device >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        print(f"Using CUDA device {args.device}")
    else:
        print("Using CPU")

    if args.inspect:
        inspect_data(args.queries, args.collection, args.mapping)

    # Build ColBERT config
    config = ColBERTConfig(
        bsize=args.bsize,
        lr=args.lr,
        warmup=args.warmup,
        doc_maxlen=args.doc_maxlen,
        dim=args.dim,
        attend_to_mask_tokens=args.attend_to_mask_tokens,
        nway=args.nway,
        accumsteps=args.accumsteps,
        similarity=args.similarity,
        use_ib_negatives=args.use_ib_negatives,
    )

    # Train
    with Run().context(RunConfig(nranks=args.nranks)):
        trainer = Trainer(
            triples=args.mapping,
            queries=args.queries,
            collection=args.collection,
            config=config,
        )
        trainer.train(checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
