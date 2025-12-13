# Fashion Embeddings Foundation

> **A comprehensive documentation repository for fashion ML projects built on Databricks**

This repository documents the foundational infrastructure for fashion AI/ML projects, including 77K+ embedded items, vector search infrastructure, and production deployment patterns.

## Overview

Two complementary fashion projects provide a robust foundation:

1. **[Fashion E-Commerce Site](https://github.com/YOUR_ORG/fashion-ecom-site)** - Production web application
2. **[Fashion Visual Search](https://github.com/YOUR_ORG/fashion-visual-search)** - Research/MVP pipeline

**Quick Stats**:
- ✅ 77,688 total embedded items (CLIP 512D)
- ✅ 44,424 product catalog with 3 embedding types per item
- ✅ 33,264 DeepFashion2 research images
- ✅ 3 Vector Search indexes (image, text, hybrid)
- ✅ Production API + React frontend
- ✅ SmolVLM attribute extraction pipeline

## What's Inside

### Core Documentation

- **[FASHION_EMBEDDINGS_FOUNDATION_SUMMARY.md](FASHION_EMBEDDINGS_FOUNDATION_SUMMARY.md)** - Complete technical overview
  - Data assets and schemas
  - Vector search infrastructure
  - Model serving configuration
  - Search capabilities
  - Performance metrics
  - Next steps for ML projects

### Key Assets Documented

**Embeddings Tables**:
- `main.fashion_demo.product_embeddings_multimodal` (44K products)
- `main.fashion_demo.fashion_items_embeddings` (33K DeepFashion2)

**Vector Search Indexes**:
- `main.fashion_demo.vs_image_search` - Visual similarity
- `main.fashion_demo.vs_text_search` - Semantic search
- `main.fashion_demo.vs_hybrid_search` - Multimodal search

**Model Serving**:
- `clip-multimodal-encoder` - CLIP ViT-B/32 endpoint

## Use Cases

This foundation enables building:

### Recommendation Systems
- Collaborative filtering
- Learning-to-Rank (LTR)
- Context-aware recommendations
- Sequential outfit building

### Computer Vision
- Style transfer
- Visual similarity search
- Attribute detection
- Outfit compatibility scoring

### NLP & Multimodal
- Product description generation
- Style explanation
- Conversational shopping assistant
- Query understanding

### Personalization
- Real-time preference learning
- Multi-armed bandits
- User embedding learning
- Behavior-based ranking

## Quick Reference

### Vector Search Query
```python
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
index = vsc.get_index(
    endpoint_name="fashion_vector_search",
    index_name="main.fashion_demo.vs_hybrid_search"
)

results = index.similarity_search(
    query_vector=embedding,
    columns=["product_id", "product_display_name", "price"],
    num_results=10
)
```

### CLIP Encoding
```python
import requests

url = f"{workspace_url}/serving-endpoints/clip-multimodal-encoder/invocations"
payload = {"dataframe_records": [{"text": "red leather jacket"}]}
response = requests.post(url, json=payload, headers=headers)
embedding = response.json()["predictions"][0]  # 512D
```

### SQL Query
```sql
-- Get products with embeddings
SELECT
    product_id,
    product_display_name,
    master_category,
    base_color,
    price
FROM main.fashion_demo.product_embeddings_multimodal
WHERE master_category = 'Apparel'
    AND price BETWEEN 50 AND 150
LIMIT 10;
```

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Total Embedded Items | 77,688 |
| Embedding Dimension | 512D (CLIP ViT-B/32) |
| Vector Search Latency | <100ms (p95) |
| Database Query Latency | ~10ms (Lakebase) |
| Model Serving Capacity | 64 concurrent requests |
| Embedding Coverage | 99.98% |

## Architecture

```
Unity Catalog (Delta Lake)
    ↓
Vector Search Indexes (3)
    ↓
Recommendation Engine
    ↓
Production API + UI
```

**Technologies**:
- Databricks Unity Catalog
- Mosaic AI Vector Search
- CLIP ViT-B/32
- FastAPI + React
- Lakebase PostgreSQL

## Getting Started

### Prerequisites
- Azure Databricks workspace
- Unity Catalog enabled
- Mosaic AI Vector Search enabled
- Access to `main.fashion_demo` catalog

### Quick Access
```python
# Unity Catalog tables
CATALOG = "main"
SCHEMA = "fashion_demo"

# Vector Search
VS_ENDPOINT = "fashion_vector_search"
VS_HYBRID_INDEX = "main.fashion_demo.vs_hybrid_search"

# Model Serving
CLIP_ENDPOINT = "clip-multimodal-encoder"
```

## Project Structure

```
fashion-embeddings-foundation/
├── README.md                                    # This file
├── FASHION_EMBEDDINGS_FOUNDATION_SUMMARY.md     # Detailed documentation
├── examples/                                    # Code examples
│   ├── vector_search_examples.py
│   ├── recommendation_examples.py
│   └── attribute_extraction_examples.py
├── schemas/                                     # Table schemas
│   ├── product_embeddings_schema.json
│   └── user_features_schema.json
└── notebooks/                                   # Reference notebooks
    ├── 01_explore_embeddings.py
    ├── 02_vector_search_demo.py
    └── 03_recommendation_demo.py
```

## Related Projects

### Source Repositories
- **[fashion-ecom-site](https://github.com/YOUR_ORG/fashion-ecom-site)** - Full-stack e-commerce application
  - Production API with FastAPI
  - React frontend
  - Lakebase PostgreSQL backend
  - SmolVLM attribute extraction

- **[fashion-visual-search](https://github.com/YOUR_ORG/fashion-visual-search)** - Research pipeline
  - 8 production notebooks
  - Python package with utilities
  - Synthetic data generation
  - Claude AI agent integration

## Data Sources

### Product Catalog
- **Dataset**: Fashion Product Images (Kaggle)
- **URL**: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
- **Size**: 44,424 products
- **Categories**: 7 master, 45 sub-categories, 143 article types

### DeepFashion2
- **Purpose**: Research and "complete the look" features
- **Size**: 33,264 images
- **Use Cases**: Outfit composition, style transfer

## Contributing

This is a documentation repository. To contribute:

1. Update documentation in `FASHION_EMBEDDINGS_FOUNDATION_SUMMARY.md`
2. Add code examples to `examples/`
3. Include reference notebooks in `notebooks/`
4. Update schemas in `schemas/` as tables evolve

## Support

For questions or issues:
- Open an issue in this repository
- Contact: kevin.ippen@databricks.com

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **Databricks** - Mosaic AI Vector Search, Model Serving, Unity Catalog
- **Anthropic** - Claude AI
- **OpenAI** - CLIP model
- **Kaggle** - Fashion Product Images dataset

---

**Last Updated**: 2025-12-13
**Status**: ✅ Production-ready foundation
**Version**: 1.0.0

🚀 **Ready to build advanced fashion AI/ML projects!**
