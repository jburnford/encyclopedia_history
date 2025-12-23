#!/usr/bin/env python3
"""
Embed encyclopedia chunks using NV-Embed-v2 on Nibi cluster.

This script loads chunks from JSONL, embeds them with NV-Embed-v2,
and saves embeddings to JSON format for later loading into Neo4j.

Usage:
    python embed_chunks_nv.py --input data/chunks_1778.jsonl --output embeddings/embeddings_1778.json

Requirements:
    - H100 80GB GPU (NV-Embed-v2 is 7B params)
    - transformers==4.46.0 (newer versions break the model)
    - sentence-transformers
    - torch with CUDA
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import numpy as np


def load_chunks(input_path: str) -> List[Dict]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def create_chunk_id(chunk: Dict) -> str:
    """Create unique chunk ID from chunk metadata."""
    headword = chunk.get('parent_headword', 'UNKNOWN')
    year = chunk.get('edition_year', 0)
    index = chunk.get('index', 0)
    return f"{headword}_{year}_{index}"


def load_model():
    """Load NV-Embed-v2 model."""
    from sentence_transformers import SentenceTransformer

    print("Loading NV-Embed-v2 model...")
    start = time.time()

    model = SentenceTransformer(
        'nvidia/NV-Embed-v2',
        trust_remote_code=True
    )
    model.max_seq_length = 32768

    print(f"Model loaded in {time.time() - start:.1f}s")
    return model


def embed_chunks(
    model,
    chunks: List[Dict],
    batch_size: int = 2
) -> Dict[str, List[float]]:
    """
    Embed chunks and return dict mapping chunk_id to embedding.

    NV-Embed-v2 produces 4096-dimensional embeddings.
    """
    texts = [c['text'] for c in chunks]
    chunk_ids = [create_chunk_id(c) for c in chunks]

    print(f"Embedding {len(texts)} chunks (batch_size={batch_size})...")
    start = time.time()

    # Embed without instruction prompt (tested: no instruction works better)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    elapsed = time.time() - start
    rate = len(texts) / elapsed if elapsed > 0 else 0
    print(f"Embedded {len(texts)} chunks in {elapsed:.1f}s ({rate:.2f} chunks/sec)")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Build result dict
    result = {}
    for chunk_id, embedding in zip(chunk_ids, embeddings):
        result[chunk_id] = embedding.tolist()

    return result


def save_embeddings(embeddings: Dict[str, List[float]], output_path: str):
    """Save embeddings to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(embeddings)} embeddings to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f)

    # Report file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved embeddings ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description='Embed encyclopedia chunks with NV-Embed-v2'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input JSONL file with chunks'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output JSON file for embeddings'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=2,
        help='Batch size for embedding (default: 2, safe for H100 80GB)'
    )
    args = parser.parse_args()

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("NV-EMBED-V2 CHUNK EMBEDDING")
    print(f"{'='*60}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Batch:  {args.batch_size}")
    print(f"{'='*60}\n")

    # Load chunks
    chunks = load_chunks(args.input)
    print(f"Loaded {len(chunks)} chunks from {input_path.name}")

    # Sample chunk info
    if chunks:
        sample = chunks[0]
        print(f"Sample chunk: {sample.get('parent_headword', '?')} "
              f"(year={sample.get('edition_year', '?')}, "
              f"chars={len(sample.get('text', ''))})")

    # Load model
    model = load_model()

    # Embed chunks
    embeddings = embed_chunks(model, chunks, args.batch_size)

    # Verify dimension
    sample_dim = len(next(iter(embeddings.values())))
    if sample_dim != 4096:
        print(f"WARNING: Expected 4096-dim embeddings, got {sample_dim}")

    # Save embeddings
    save_embeddings(embeddings, args.output)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Chunks embedded: {len(embeddings)}")
    print(f"Embedding dim:   {sample_dim}")
    print(f"Output file:     {args.output}")


if __name__ == '__main__':
    main()
