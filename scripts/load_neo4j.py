#!/usr/bin/env python3
"""
Load encyclopedia chunks and embeddings into Neo4j.

Creates a knowledge graph with:
- EB_Edition nodes (year, title)
- EB_Article nodes (headword, edition_year)
- EB_Chunk nodes (text, embedding, section_title)

And relationships:
- (Edition)-[:CONTAINS]->(Article)
- (Article)-[:HAS_CHUNK]->(Chunk)

Usage:
    python load_neo4j.py --chunks data/chunks_1778.jsonl --embeddings embeddings/embeddings_1778.json

    # Load multiple editions:
    python load_neo4j.py --chunks data/chunks_1778.jsonl --embeddings embeddings/embeddings_1778.json
    python load_neo4j.py --chunks data/chunks_1823.jsonl --embeddings embeddings/embeddings_1823.json

Environment:
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env file or environment
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j package not installed. Run: pip install neo4j")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


# Edition titles (can be extended)
EDITION_TITLES = {
    1778: "Encyclopaedia Britannica, 2nd Edition",
    1823: "Encyclopaedia Britannica, 6th Edition",
}

# Embedding dimension for NV-Embed-v2
EMBEDDING_DIM = 4096


def get_neo4j_connection() -> Tuple[str, str, str]:
    """Get Neo4j connection details from environment."""
    uri = os.getenv('NEO4J_URI', 'bolt://206.12.90.118:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD')

    if not password:
        print("ERROR: NEO4J_PASSWORD not set in environment or .env file")
        sys.exit(1)

    return uri, user, password


def load_chunks(chunks_path: str) -> List[Dict]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def load_embeddings(embeddings_path: str) -> Dict[str, List[float]]:
    """Load embeddings from JSON file."""
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_chunk_id(chunk: Dict) -> str:
    """Create unique chunk ID from chunk metadata."""
    headword = chunk.get('parent_headword', 'UNKNOWN')
    year = chunk.get('edition_year', 0)
    index = chunk.get('index', 0)
    return f"{headword}_{year}_{index}"


class Neo4jLoader:
    """Load encyclopedia data into Neo4j."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._verify_connection()

    def _verify_connection(self):
        """Verify Neo4j connection."""
        with self.driver.session() as session:
            result = session.run("RETURN 1 as n")
            result.single()
        print("Connected to Neo4j successfully")

    def close(self):
        """Close the driver."""
        self.driver.close()

    def setup_constraints(self):
        """Create constraints and indexes."""
        constraints = [
            # Unique constraints
            ("eb_edition_year", "CREATE CONSTRAINT eb_edition_year IF NOT EXISTS FOR (e:EB_Edition) REQUIRE e.year IS UNIQUE"),
            ("eb_article_id", "CREATE CONSTRAINT eb_article_id IF NOT EXISTS FOR (a:EB_Article) REQUIRE (a.headword, a.edition_year) IS UNIQUE"),
            ("eb_chunk_id", "CREATE CONSTRAINT eb_chunk_id IF NOT EXISTS FOR (c:EB_Chunk) REQUIRE c.chunk_id IS UNIQUE"),
        ]

        with self.driver.session() as session:
            for name, query in constraints:
                try:
                    session.run(query)
                    print(f"  Created constraint: {name}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print(f"  Constraint exists: {name}")
                    else:
                        print(f"  Warning on {name}: {e}")

    def setup_vector_index(self):
        """Create vector index for chunk embeddings."""
        # Check if index already exists
        with self.driver.session() as session:
            result = session.run("SHOW INDEXES WHERE name = 'eb_chunk_embedding'")
            if result.single():
                print("  Vector index 'eb_chunk_embedding' already exists")
                return

        # Create vector index
        query = """
        CREATE VECTOR INDEX eb_chunk_embedding IF NOT EXISTS
        FOR (c:EB_Chunk) ON c.embedding
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: $dim,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
        with self.driver.session() as session:
            session.run(query, dim=EMBEDDING_DIM)
            print(f"  Created vector index (dim={EMBEDDING_DIM})")

    def create_edition(self, year: int, title: str = None):
        """Create or get an edition node."""
        if title is None:
            title = EDITION_TITLES.get(year, f"Encyclopaedia Britannica ({year})")

        query = """
        MERGE (e:EB_Edition {year: $year})
        ON CREATE SET e.title = $title
        RETURN e
        """
        with self.driver.session() as session:
            session.run(query, year=year, title=title)

    def load_chunks_batch(
        self,
        chunks: List[Dict],
        embeddings: Dict[str, List[float]],
        batch_size: int = 100
    ):
        """Load chunks in batches for performance."""
        # Group chunks by article
        articles = defaultdict(list)
        for chunk in chunks:
            headword = chunk.get('parent_headword', 'UNKNOWN')
            year = chunk.get('edition_year', 0)
            key = (headword, year)
            articles[key].append(chunk)

        # Create edition first
        if chunks:
            year = chunks[0].get('edition_year', 0)
            self.create_edition(year)
            print(f"  Created/verified edition: {year}")

        # Process in batches
        chunk_batch = []
        missing_embeddings = 0

        for chunk in tqdm(chunks, desc="Preparing chunks"):
            chunk_id = create_chunk_id(chunk)
            embedding = embeddings.get(chunk_id)

            if embedding is None:
                missing_embeddings += 1
                continue

            chunk_batch.append({
                'chunk_id': chunk_id,
                'text': chunk.get('text', ''),
                'headword': chunk.get('parent_headword', 'UNKNOWN'),
                'edition_year': chunk.get('edition_year', 0),
                'chunk_index': chunk.get('index', 0),
                'section_title': chunk.get('section_title', ''),
                'section_index': chunk.get('section_index', 0),
                'char_start': chunk.get('char_start', 0),
                'char_end': chunk.get('char_end', 0),
                'embedding': embedding
            })

            if len(chunk_batch) >= batch_size:
                self._insert_chunk_batch(chunk_batch)
                chunk_batch = []

        # Insert remaining
        if chunk_batch:
            self._insert_chunk_batch(chunk_batch)

        if missing_embeddings > 0:
            print(f"  Warning: {missing_embeddings} chunks missing embeddings")

        # Create article nodes and relationships
        print("  Creating article nodes and relationships...")
        self._create_articles_and_relationships(articles)

    def _insert_chunk_batch(self, batch: List[Dict]):
        """Insert a batch of chunks."""
        query = """
        UNWIND $chunks as chunk
        MERGE (c:EB_Chunk {chunk_id: chunk.chunk_id})
        SET c.text = chunk.text,
            c.headword = chunk.headword,
            c.edition_year = chunk.edition_year,
            c.chunk_index = chunk.chunk_index,
            c.section_title = chunk.section_title,
            c.section_index = chunk.section_index,
            c.char_start = chunk.char_start,
            c.char_end = chunk.char_end,
            c.embedding = chunk.embedding
        """
        with self.driver.session() as session:
            session.run(query, chunks=batch)

    def _create_articles_and_relationships(self, articles: Dict):
        """Create article nodes and relationships."""
        with self.driver.session() as session:
            for (headword, year), chunks in tqdm(articles.items(), desc="Creating articles"):
                # Create article node
                article_query = """
                MERGE (a:EB_Article {headword: $headword, edition_year: $year})
                SET a.chunk_count = $chunk_count
                """
                session.run(
                    article_query,
                    headword=headword,
                    year=year,
                    chunk_count=len(chunks)
                )

                # Link edition to article
                edition_link = """
                MATCH (e:EB_Edition {year: $year})
                MATCH (a:EB_Article {headword: $headword, edition_year: $year})
                MERGE (e)-[:CONTAINS]->(a)
                """
                session.run(edition_link, headword=headword, year=year)

                # Link article to chunks
                chunk_link = """
                MATCH (a:EB_Article {headword: $headword, edition_year: $year})
                MATCH (c:EB_Chunk)
                WHERE c.headword = $headword AND c.edition_year = $year
                MERGE (a)-[:HAS_CHUNK]->(c)
                """
                session.run(chunk_link, headword=headword, year=year)

    def get_stats(self) -> Dict:
        """Get database statistics."""
        with self.driver.session() as session:
            stats = {}

            # Count nodes
            for label in ['EB_Edition', 'EB_Article', 'EB_Chunk']:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                stats[label] = result.single()['count']

            # Count relationships
            result = session.run("MATCH ()-[r:CONTAINS]->() RETURN count(r) as count")
            stats['CONTAINS'] = result.single()['count']

            result = session.run("MATCH ()-[r:HAS_CHUNK]->() RETURN count(r) as count")
            stats['HAS_CHUNK'] = result.single()['count']

            return stats


def main():
    parser = argparse.ArgumentParser(
        description='Load encyclopedia chunks and embeddings into Neo4j'
    )
    parser.add_argument(
        '--chunks', '-c',
        type=str,
        required=True,
        help='Input JSONL file with chunks'
    )
    parser.add_argument(
        '--embeddings', '-e',
        type=str,
        required=True,
        help='Input JSON file with embeddings'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=100,
        help='Batch size for Neo4j inserts (default: 100)'
    )
    parser.add_argument(
        '--setup-only',
        action='store_true',
        help='Only create constraints and indexes, don\'t load data'
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("ENCYCLOPEDIA NEO4J LOADER")
    print(f"{'='*60}")

    # Get connection details
    uri, user, password = get_neo4j_connection()
    print(f"Neo4j URI: {uri}")
    print(f"Neo4j User: {user}")

    # Check input files
    if not args.setup_only:
        for path, name in [(args.chunks, 'chunks'), (args.embeddings, 'embeddings')]:
            if not Path(path).exists():
                print(f"ERROR: {name} file not found: {path}")
                sys.exit(1)

    print(f"Chunks: {args.chunks}")
    print(f"Embeddings: {args.embeddings}")
    print(f"{'='*60}\n")

    # Connect to Neo4j
    loader = Neo4jLoader(uri, user, password)

    try:
        # Setup constraints and indexes
        print("Setting up constraints...")
        loader.setup_constraints()

        print("\nSetting up vector index...")
        loader.setup_vector_index()

        if args.setup_only:
            print("\nSetup complete (--setup-only flag set)")
            return

        # Load data
        print("\nLoading chunks...")
        chunks = load_chunks(args.chunks)
        print(f"  Loaded {len(chunks)} chunks")

        print("\nLoading embeddings...")
        embeddings = load_embeddings(args.embeddings)
        print(f"  Loaded {len(embeddings)} embeddings")

        # Verify embedding dimension
        sample_emb = next(iter(embeddings.values()))
        if len(sample_emb) != EMBEDDING_DIM:
            print(f"WARNING: Expected {EMBEDDING_DIM}-dim embeddings, got {len(sample_emb)}")

        print("\nLoading into Neo4j...")
        loader.load_chunks_batch(chunks, embeddings, args.batch_size)

        # Print stats
        print("\nDatabase statistics:")
        stats = loader.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value:,}")

        print(f"\n{'='*60}")
        print("LOAD COMPLETE")
        print(f"{'='*60}")
        print("\nTest with:")
        print("""
  # Find chunks by article
  MATCH (a:EB_Article {headword: 'CHEMISTRY'})-[:HAS_CHUNK]->(c:EB_Chunk)
  RETURN c.chunk_id, c.section_title, left(c.text, 100)
  LIMIT 5

  # Vector similarity search (requires query embedding)
  CALL db.index.vector.queryNodes('eb_chunk_embedding', 5, $query_embedding)
  YIELD node, score
  RETURN node.headword, node.section_title, score
""")

    finally:
        loader.close()


if __name__ == '__main__':
    main()
