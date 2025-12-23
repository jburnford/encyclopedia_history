#!/usr/bin/env python3
"""
Test NV-Embed-v2 on historical encyclopedia articles.

This script tests the hypothesis that instruction-tuned embeddings
can better handle semantic drift in historical text.
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Import chunking utilities
from chunk_articles import (
    load_and_chunk, Chunk, Article,
    LEATHER_QUERIES, SEMANTIC_DRIFT_QUERIES, LONG_ARTICLE_QUERIES,
    LEATHER_ARTICLES, SEMANTIC_DRIFT_ARTICLES, LONG_ARTICLES
)

# Historical instruction for NV-Embed-v2
HISTORICAL_INSTRUCTION = """Given a query about 18th-century knowledge from the 1815 Encyclopaedia Britannica, retrieve relevant passages. Note that historical terminology may differ from modern usage - for example, 'phlogiston' was the accepted theory of combustion, 'physics' often meant medicine, and 'broadcast' meant scattering seeds."""


@dataclass
class QueryResult:
    """Result for a single query."""
    query: str
    expected: List[str]
    retrieved: List[str]
    scores: List[float]
    rank: int  # Rank of first relevant result (0 if not found)
    mrr: float  # 1/rank or 0


@dataclass
class TestResults:
    """Complete test results."""
    model: str
    instruction: str
    num_articles: int
    num_chunks: int
    embed_time: float
    query_time: float
    queries: List[Dict]
    mrr: float
    recall_at_5: float
    recall_at_10: float


def load_model():
    """Load NV-Embed-v2 model."""
    from sentence_transformers import SentenceTransformer

    print("Loading NV-Embed-v2...")
    start = time.time()

    model = SentenceTransformer(
        'nvidia/NV-Embed-v2',
        trust_remote_code=True
    )
    model.max_seq_length = 32768

    print(f"  Model loaded in {time.time() - start:.1f}s")
    return model


def embed_documents(model, chunks: List[Chunk], batch_size: int = 32) -> np.ndarray:
    """Embed document chunks (no instruction needed)."""
    texts = [c.text for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    elapsed = time.time() - start
    print(f"  Embedded in {elapsed:.1f}s ({len(texts)/elapsed:.1f} chunks/sec)")

    return np.array(embeddings), elapsed


def embed_queries(
    model,
    queries: List[str],
    instruction: str = None
) -> np.ndarray:
    """Embed queries with optional instruction."""
    print(f"Embedding {len(queries)} queries...")
    start = time.time()

    if instruction:
        embeddings = model.encode(
            queries,
            prompt=instruction,
            normalize_embeddings=True
        )
    else:
        embeddings = model.encode(
            queries,
            normalize_embeddings=True
        )

    elapsed = time.time() - start
    print(f"  Embedded in {elapsed:.1f}s")

    return np.array(embeddings), elapsed


def cosine_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity (assumes normalized vectors)."""
    return np.dot(doc_embs, query_emb)


def evaluate_query(
    query: str,
    expected: List[str],
    query_emb: np.ndarray,
    doc_embs: np.ndarray,
    chunks: List[Chunk],
    k: int = 10
) -> QueryResult:
    """Evaluate a single query."""
    # Compute similarities
    similarities = cosine_similarity(query_emb, doc_embs)

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:k]
    retrieved = [chunks[i].headword for i in top_indices]
    scores = [float(similarities[i]) for i in top_indices]

    # Find rank of first relevant result
    expected_upper = {e.upper() for e in expected}
    rank = 0
    for i, hw in enumerate(retrieved, 1):
        if hw.upper() in expected_upper:
            rank = i
            break

    mrr = 1.0 / rank if rank > 0 else 0.0

    return QueryResult(
        query=query,
        expected=expected,
        retrieved=retrieved,
        scores=scores,
        rank=rank,
        mrr=mrr
    )


def run_evaluation(
    model,
    chunks: List[Chunk],
    doc_embeddings: np.ndarray,
    queries: List[Tuple[str, List[str]]],
    instruction: str = None,
    query_type: str = "unknown"
) -> Tuple[List[QueryResult], float]:
    """Run evaluation on a set of queries."""
    query_texts = [q[0] for q in queries]
    query_embs, query_time = embed_queries(model, query_texts, instruction)

    results = []
    for i, (query_text, expected) in enumerate(queries):
        result = evaluate_query(
            query_text, expected,
            query_embs[i], doc_embeddings, chunks
        )
        results.append(result)

    return results, query_time


def main():
    parser = argparse.ArgumentParser(description='Test NV-Embed-v2 on encyclopedia articles')
    parser.add_argument('--data', type=str, default='data/articles_1815_clean.jsonl',
                        help='Path to articles JSONL file')
    parser.add_argument('--output', type=str, default='results/nv_embed_results.json',
                        help='Output path for results JSON')
    parser.add_argument('--instruction', type=str, default=HISTORICAL_INSTRUCTION,
                        help='Instruction prompt for queries')
    parser.add_argument('--no-instruction', action='store_true',
                        help='Run without instruction (for comparison)')
    parser.add_argument('--chunk-size', type=int, default=800,
                        help='Chunk size in characters')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for embedding')
    parser.add_argument('--test-type', type=str, default='all',
                        choices=['leather', 'semantic', 'long', 'all'],
                        help='Type of test to run')
    args = parser.parse_args()

    # Select target articles based on test type
    if args.test_type == 'leather':
        target_articles = LEATHER_ARTICLES
        queries = LEATHER_QUERIES
    elif args.test_type == 'semantic':
        target_articles = SEMANTIC_DRIFT_ARTICLES + LEATHER_ARTICLES[:5]
        queries = SEMANTIC_DRIFT_QUERIES
    elif args.test_type == 'long':
        target_articles = LONG_ARTICLES
        queries = LONG_ARTICLE_QUERIES
    else:  # all
        target_articles = list(set(
            LEATHER_ARTICLES + SEMANTIC_DRIFT_ARTICLES + LONG_ARTICLES
        ))
        queries = LEATHER_QUERIES + SEMANTIC_DRIFT_QUERIES + LONG_ARTICLE_QUERIES

    # Load and chunk articles
    print(f"\n{'='*60}")
    print(f"NV-EMBED-V2 EMBEDDING TEST")
    print(f"{'='*60}")
    print(f"\nTest type: {args.test_type}")
    print(f"Target articles: {len(target_articles)}")

    articles, chunks = load_and_chunk(
        args.data,
        headwords=target_articles,
        chunk_size=args.chunk_size
    )

    print(f"\nLoaded {len(articles)} articles, {len(chunks)} chunks")
    for a in articles[:10]:
        print(f"  {a.headword}: {len(a.text):,} chars")
    if len(articles) > 10:
        print(f"  ... and {len(articles) - 10} more")

    # Load model
    model = load_model()

    # Embed documents
    doc_embeddings, embed_time = embed_documents(model, chunks, args.batch_size)

    # Run evaluation
    instruction = None if args.no_instruction else args.instruction
    print(f"\nInstruction: {'None' if instruction is None else instruction[:50] + '...'}")

    all_results = []
    total_query_time = 0

    # Evaluate each query type separately for analysis
    query_groups = [
        ('leather', LEATHER_QUERIES),
        ('semantic_drift', SEMANTIC_DRIFT_QUERIES),
        ('long_article', LONG_ARTICLE_QUERIES),
    ]

    for group_name, group_queries in query_groups:
        if args.test_type != 'all' and args.test_type != group_name.split('_')[0]:
            continue

        print(f"\n--- {group_name} queries ---")
        results, query_time = run_evaluation(
            model, chunks, doc_embeddings,
            group_queries, instruction, group_name
        )
        total_query_time += query_time

        for r in results:
            r_dict = asdict(r)
            r_dict['query_type'] = group_name
            all_results.append(r_dict)

            status = "HIT" if r.rank > 0 else "MISS"
            print(f"  [{status}] {r.query[:40]}...")
            print(f"       Expected: {r.expected}, Got: {r.retrieved[:3]}, Rank: {r.rank}")

    # Compute overall metrics
    mrr = np.mean([r['mrr'] for r in all_results])
    recall_5 = np.mean([1 if r['rank'] <= 5 and r['rank'] > 0 else 0 for r in all_results])
    recall_10 = np.mean([1 if r['rank'] <= 10 and r['rank'] > 0 else 0 for r in all_results])

    # Results by query type
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    for group_name in ['leather', 'semantic_drift', 'long_article']:
        group_results = [r for r in all_results if r['query_type'] == group_name]
        if group_results:
            group_mrr = np.mean([r['mrr'] for r in group_results])
            print(f"\n{group_name}:")
            print(f"  MRR: {group_mrr:.3f}")
            print(f"  Queries: {len(group_results)}")

    print(f"\n{'='*60}")
    print(f"OVERALL:")
    print(f"  MRR: {mrr:.3f}")
    print(f"  Recall@5: {recall_5:.3f}")
    print(f"  Recall@10: {recall_10:.3f}")
    print(f"  Embed time: {embed_time:.1f}s")
    print(f"  Query time: {total_query_time:.1f}s")
    print(f"{'='*60}")

    # Save results
    output = TestResults(
        model="NV-Embed-v2",
        instruction=instruction or "none",
        num_articles=len(articles),
        num_chunks=len(chunks),
        embed_time=embed_time,
        query_time=total_query_time,
        queries=all_results,
        mrr=mrr,
        recall_at_5=recall_5,
        recall_at_10=recall_10
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(asdict(output), f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
