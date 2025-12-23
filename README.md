# Encyclopedia History Embedding Test

Comparing **NV-Embed-v2** (instruction-tuned) vs **voyage-3** (API) for historical encyclopedia retrieval.

## Project Goal

Test whether instruction-tuned embedding models can better handle **semantic drift** in historical text (1815 Encyclopaedia Britannica) by providing context like:

> "Given a query about 18th-century scientific knowledge, retrieve relevant passages from the 1815 Encyclopaedia Britannica."

## Models Compared

| Model | MTEB Score | Instruction Tuning | Deployment |
|-------|------------|-------------------|------------|
| voyage-3 | ~69 | No | API |
| NV-Embed-v2 | 72.31 (#1) | Yes | Self-hosted (H100) |

## Quick Start

### Local (voyage-3 baseline)
```bash
pip install -r requirements.txt
cp .env.example .env  # Add your Voyage API key
python scripts/test_voyage.py
```

### Nibi Cluster (NV-Embed-v2)
```bash
# One-time setup
sbatch slurm/setup_env.sh
sbatch slurm/download_model.sh

# Run test
sbatch slurm/run_nv_embed_test.sh
```

## Test Queries

### Semantic Drift Queries
Words whose meanings have changed since 1815:
- "combustion" → Should find PHLOGISTON (not oxygen chemistry)
- "broadcast" → Should find agriculture (not media)
- "atom" → Should find philosophy (not nuclear physics)

### Domain Queries (Leather/Tanning)
- "vegetable tanning oak bark process"
- "hide preparation lime soaking"
- "currying finishing leather oil"

## Results

See `results/` for JSON output files and comparison reports.

## Data

- **Source**: 1815 Encyclopaedia Britannica (NLS digitization)
- **Articles**: 18,172
- **Format**: JSONL with `headword` and `text` fields

## License

- Code: MIT
- NV-Embed-v2: CC-BY-NC-4.0 (non-commercial use only)
