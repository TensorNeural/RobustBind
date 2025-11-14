#!/usr/bin/env python3
import sys
import traceback
import argparse
sys.path.insert(0, '/workspace/RobustBind')
from transfer_attack import main

ns = argparse.Namespace(
    modality='image',
    val_json=None,
    dataset_root=None,
    centre_embeddings=None,
    unibind_weights=None,
    robust_lora_weights=None,
    src_model='unibind',
    run_all=False,
    run_max_samples=2,
    eps=2.0,
    steps=1,
    batch_size=1,
    max_samples=2,
    output='/data/output/transfer_attack_test'
)

def _run():
    try:
        main(ns)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == '__main__':
    _run()
