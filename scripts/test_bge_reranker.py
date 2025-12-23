#!/usr/bin/env python3
"""
Test BGE-Reranker-v2.5 on encyclopedia articles.

Two-stage retrieval:
1. Fast bi-encoder retrieval (NV-Embed-v2) → top-k candidates
2. Cross-encoder reranking (BGE-Reranker) → final ranking

Tests hypothesis: reranking improves precision on semantic drift queries.
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
    load_and_chunk, Chunk,
    LEATHER_QUERIES, SEMANTIC_DRIFT_QUERIES, LONG_ARTICLE_QUERIES,
    LEATHER_ARTICLES, SEMANTIC_DRIFT_ARTICLES, LONG_ARTICLES
)


@dataclass
class RerankerResult:
    """Result comparing retrieval with and without reranking."""
    query: str
    expected: List[str]
    retrieval_rank: int      # Rank before reranking
    reranked_rank: int       # Rank after reranking
    retrieval_top5: List[str]
    reranked_top5: List[str]
    improvement: int         # Positive = reranking helped


@dataclass
class TestResults:
    """Complete test results."""
    retriever: str
    reranker: str
    num_chunks: int
    retrieval_time: float
    rerank_time: float
    results: List[Dict]
    retrieval_mrr: float
    reranked_mrr: float
    mrr_improvement: float


def load_retriever(model_name: str = 'nvidia/NV-Embed-v2'):
    """Load the retrieval model."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading retriever: {model_name}...")
    start = time.time()

    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.max_seq_length = 32768

    print(f"  Loaded in {time.time() - start:.1f}s")
    return model


def load_reranker(model_name: str = 'BAAI/bge-reranker-v2.5-gemma2-lightweight',
                  use_fp16: bool = True,
                  use_full_layers: bool = True,
                  cutoff_layers: List[int] = None):
    """Load the reranker model.

    Args:
        model_name: HuggingFace model name
        use_fp16: Use FP16 for faster computation
        use_full_layers: If True, use all 42 layers (requires ~40GB VRAM)
                        If False, use cutoff_layers for lightweight mode
        cutoff_layers: Which layers to use (only if use_full_layers=False)
    """
    from FlagEmbedding import FlagLLMReranker, LayerWiseFlagLLMReranker

    print(f"Loading reranker: {model_name}...")
    start = time.time()

    if use_full_layers:
        # Full model - all layers, maximum accuracy
        # Requires ~40GB VRAM in FP16
        reranker = FlagLLMReranker(
            model_name,
            use_fp16=use_fp16
        )
        print(f"  Mode: FULL (all layers)")
    else:
        # Lightweight mode - layer cutoff for speed
        if cutoff_layers is None:
            cutoff_layers = [8]  # Default to first 8 layers
        reranker = LayerWiseFlagLLMReranker(
            model_name,
            use_fp16=use_fp16,
            cutoff_layers=cutoff_layers
        )
        print(f"  Mode: LIGHTWEIGHT (cutoff_layers={cutoff_layers})")

    print(f"  Loaded in {time.time() - start:.1f}s")
    return reranker


def retrieve_candidates(
    retriever,
    query: str,
    doc_embeddings: np.ndarray,
    chunks: List[Chunk],
    top_k: int = 50
) -> List[Tuple[Chunk, float]]:
    """Retrieve top-k candidates using bi-encoder."""
    # Embed query
    query_emb = retriever.encode([query], normalize_embeddings=True)[0]

    # Compute similarities
    similarities = np.dot(doc_embeddings, query_emb)

    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [(chunks[i], float(similarities[i])) for i in top_indices]


def rerank_candidates(
    reranker,
    query: str,
    candidates: List[Tuple[Chunk, float]]
) -> List[Tuple[Chunk, float]]:
    """Rerank candidates using cross-encoder."""
    # Prepare pairs for reranking
    pairs = [[query, c[0].text] for c in candidates]

    # Get reranker scores
    scores = reranker.compute_score(pairs)

    # Sort by reranker score
    reranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [(c[0], score) for (c, _), score in reranked]


def find_rank(results: List[Tuple[Chunk, float]], expected: List[str]) -> int:
    """Find rank of first expected article in results."""
    expected_upper = {e.upper() for e in expected}
    for i, (chunk, _) in enumerate(results, 1):
        if chunk.headword.upper() in expected_upper:
            return i
    return 0


def evaluate_queries(
    retriever,
    reranker,
    queries: List[Tuple[str, List[str]]],
    doc_embeddings: np.ndarray,
    chunks: List[Chunk],
    retrieve_k: int = 50
) -> Tuple[List[RerankerResult], float, float]:
    """Evaluate queries with and without reranking."""
    results = []
    total_retrieval_time = 0
    total_rerank_time = 0

    for query, expected in tqdm(queries, desc="Evaluating"):
        # Stage 1: Retrieval
        start = time.time()
        candidates = retrieve_candidates(
            retriever, query, doc_embeddings, chunks, top_k=retrieve_k
        )
        total_retrieval_time += time.time() - start

        retrieval_rank = find_rank(candidates, expected)
        retrieval_top5 = [c[0].headword for c in candidates[:5]]

        # Stage 2: Reranking
        start = time.time()
        reranked = rerank_candidates(reranker, query, candidates)
        total_rerank_time += time.time() - start

        reranked_rank = find_rank(reranked, expected)
        reranked_top5 = [c[0].headword for c in reranked[:5]]

        results.append(RerankerResult(
            query=query,
            expected=expected,
            retrieval_rank=retrieval_rank,
            reranked_rank=reranked_rank,
            retrieval_top5=retrieval_top5,
            reranked_top5=reranked_top5,
            improvement=retrieval_rank - reranked_rank if reranked_rank > 0 else 0
        ))

    return results, total_retrieval_time, total_rerank_time


def main():
    parser = argparse.ArgumentParser(description='Test BGE-Reranker on encyclopedia')
    parser.add_argument('--data', type=str, default='data/articles_1815_clean.jsonl',
                        help='Path to articles JSONL')
    parser.add_argument('--output', type=str, default='results/reranker_results.json',
                        help='Output path')
    parser.add_argument('--retriever', type=str, default='nvidia/NV-Embed-v2',
                        help='Retriever model')
    parser.add_argument('--reranker', type=str,
                        default='BAAI/bge-reranker-v2.5-gemma2-lightweight',
                        help='Reranker model')
    parser.add_argument('--full-layers', action='store_true', default=True,
                        help='Use all layers (full power, requires ~40GB VRAM)')
    parser.add_argument('--lightweight', action='store_true',
                        help='Use lightweight mode with layer cutoffs')
    parser.add_argument('--cutoff-layers', type=int, nargs='+', default=[8],
                        help='Cutoff layers for lightweight mode')
    parser.add_argument('--retrieve-k', type=int, default=50,
                        help='Number of candidates to retrieve')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for embedding')
    parser.add_argument('--test-type', type=str, default='all',
                        choices=['leather', 'semantic', 'long', 'all'])
    args = parser.parse_args()

    # Select articles and queries
    if args.test_type == 'leather':
        target_articles = LEATHER_ARTICLES
        queries = LEATHER_QUERIES
    elif args.test_type == 'semantic':
        target_articles = SEMANTIC_DRIFT_ARTICLES + LEATHER_ARTICLES[:5]
        queries = SEMANTIC_DRIFT_QUERIES
    elif args.test_type == 'long':
        target_articles = LONG_ARTICLES
        queries = LONG_ARTICLE_QUERIES
    else:
        target_articles = list(set(
            LEATHER_ARTICLES + SEMANTIC_DRIFT_ARTICLES + LONG_ARTICLES
        ))
        queries = LEATHER_QUERIES + SEMANTIC_DRIFT_QUERIES + LONG_ARTICLE_QUERIES

    # Load data
    print(f"\n{'='*60}")
    print("BGE-RERANKER TEST")
    print(f"{'='*60}")

    articles, chunks = load_and_chunk(args.data, headwords=target_articles)
    print(f"\nLoaded {len(articles)} articles, {len(chunks)} chunks")

    # Load models
    retriever = load_retriever(args.retriever)
    use_full = not args.lightweight  # Full layers by default unless --lightweight
    reranker = load_reranker(
        args.reranker,
        use_full_layers=use_full,
        cutoff_layers=args.cutoff_layers if args.lightweight else None
    )

    # Embed documents
    print(f"\nEmbedding {len(chunks)} chunks...")
    start = time.time()
    texts = [c.text for c in chunks]
    doc_embeddings = retriever.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    embed_time = time.time() - start
    print(f"  Embedded in {embed_time:.1f}s")

    # Evaluate
    results, retrieval_time, rerank_time = evaluate_queries(
        retriever, reranker, queries, doc_embeddings, chunks, args.retrieve_k
    )

    # Compute metrics
    retrieval_mrr = np.mean([1/r.retrieval_rank if r.retrieval_rank > 0 else 0 for r in results])
    reranked_mrr = np.mean([1/r.reranked_rank if r.reranked_rank > 0 else 0 for r in results])

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    for r in results:
        status = "↑" if r.improvement > 0 else ("↓" if r.improvement < 0 else "=")
        print(f"\n{r.query[:50]}...")
        print(f"  Expected: {r.expected}")
        print(f"  Retrieval rank: {r.retrieval_rank} → Reranked: {r.reranked_rank} [{status}]")
        print(f"  Retrieval top-5: {r.retrieval_top5}")
        print(f"  Reranked top-5:  {r.reranked_top5}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Retrieval MRR:  {retrieval_mrr:.3f}")
    print(f"  Reranked MRR:   {reranked_mrr:.3f}")
    print(f"  Improvement:    {(reranked_mrr - retrieval_mrr)*100:+.1f}%")
    print(f"  Retrieval time: {retrieval_time:.2f}s")
    print(f"  Rerank time:    {rerank_time:.2f}s")

    # Save results
    output = TestResults(
        retriever=args.retriever,
        reranker=args.reranker,
        num_chunks=len(chunks),
        retrieval_time=retrieval_time,
        rerank_time=rerank_time,
        results=[asdict(r) for r in results],
        retrieval_mrr=retrieval_mrr,
        reranked_mrr=reranked_mrr,
        mrr_improvement=reranked_mrr - retrieval_mrr
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(asdict(output), f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
