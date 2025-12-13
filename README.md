# Graph-Based Outfit Composition System

> **Build intelligent outfit recommendations using graph neural networks and fashion embeddings**

A complete implementation guide for building a production-ready outfit recommendation system on Azure Databricks using:
- 22,000 DeepFashion2 outfit images (co-occurrence patterns)
- 44,000 product catalog items (visual similarity)
- Graph-based recommendations (Polyvore-style approach)
- Databricks Vector Search & Unity Catalog

## Project Status

**Current Status**: ✅ Ready to Build
**Data Quality**: 22,000 clean DeepFashion2 embeddings (100% valid)
**Implementation Time**: ~3 weeks (20-25 hours total)

### Data Assets ✅

| Asset | Table | Count | Status |
|-------|-------|-------|--------|
| DeepFashion2 embeddings | `main.fashion_demo.deepfashion2_clip_embeddings` | 22,000 | ✅ Validated |
| Product embeddings | `main.fashion_demo.product_embeddings_multimodal` | 44,424 | ✅ Available |
| Product catalog | `main.fashion_demo.products` | 44,424 | ✅ Available |
| Vector Search | 3 indexes (image/text/hybrid) | - | ✅ Operational |

## What You'll Build

### Primary Features

1. **"Complete the Outfit"** - Given 1-2 items → suggest complementary items
2. **Outfit Compatibility Scoring** - Rate if items go well together (0-1 score)
3. **"Shop the Look"** - Upload outfit image → find products to recreate it

### Expected Performance

| Metric | Target |
|--------|--------|
| Fill-in-blank accuracy | 50-65% |
| Compatibility AUC | 0.75-0.85 |
| API latency | <200ms |
| Coverage | 70-85% of products |

## Implementation Phases

This project is broken into **5 sequential phases**, each with a complete implementation guide:

### Phase 1: Data Validation (2-3 hours) ✅
**Guide**: [DATA_VALIDATION_GUIDE.md](DATA_VALIDATION_GUIDE.md)

Validate data quality and create working tables:
- Verify 22K DeepFashion2 embeddings
- Test Vector Search integration
- Create working datasets
- Export product taxonomy

**Start here!** All validation steps are documented.

### Phase 2: Vector Search Mapping (3-4 hours)
**Guide**: [VECTOR_SEARCH_MAPPING_GUIDE.md](VECTOR_SEARCH_MAPPING_GUIDE.md)

Map DeepFashion2 items to product catalog:
- Top-5 similarity matching
- Quality control & validation
- Category alignment checks
- Store mappings in Delta table

### Phase 3-5: Graph, Recommendations, API (15-20 hours)
**Guide**: [PHASES_3_4_5_BRIEF.md](PHASES_3_4_5_BRIEF.md)

- **Phase 3**: Build NetworkX graph with 100K+ edges
- **Phase 4**: Pre-compute recommendations and evaluate
- **Phase 5**: Deploy 3 API endpoints to production

Full detailed guides provided after completing earlier phases.

## Quick Reference

### Key Resources

**Documentation**:
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - Complete project plan with timeline
- [DATA_VALIDATION_GUIDE.md](DATA_VALIDATION_GUIDE.md) - Phase 1 step-by-step guide
- [VECTOR_SEARCH_MAPPING_GUIDE.md](VECTOR_SEARCH_MAPPING_GUIDE.md) - Phase 2 step-by-step guide
- [queries/export_taxonomy.sql](queries/export_taxonomy.sql) - Taxonomy mapping queries
- [examples/vector_search_examples.py](examples/vector_search_examples.py) - Python code samples

**Data Tables**:
- `main.fashion_demo.products` - Product catalog (44K items)
- `main.fashion_demo.product_embeddings_multimodal` - Product embeddings
- `main.fashion_demo.deepfashion2_clip_embeddings` - DeepFashion2 embeddings (22K items)

**Infrastructure**:
- Vector Search Endpoint: `fashion_vector_search`
- CLIP Model Endpoint: `clip-multimodal-encoder`
- Unity Catalog: `main.fashion_demo`

## Architecture

```
DeepFashion2 Outfits (22K)     Product Catalog (44K)
         ↓                              ↓
    CLIP Embeddings              CLIP Embeddings
         ↓                              ↓
         └──────── Vector Search ───────┘
                        ↓
              Graph Construction
           (co-occurrence + similarity)
                        ↓
         Recommendation Engine (NetworkX)
                        ↓
              Pre-computed Results
                        ↓
         FastAPI Endpoints (<200ms)
```

## Getting Started

### Prerequisites

**Access Required**:
- ✅ Azure Databricks workspace
- ✅ Unity Catalog: `main.fashion_demo` schema
- ✅ Vector Search endpoint: `fashion_vector_search`
- ✅ CLIP model serving endpoint
- ✅ Databricks cluster (Standard_DS3_v2 or better)

**Skills Required**:
- **Python**: Intermediate (PySpark, NetworkX)
- **SQL**: Basic (SELECT, JOIN, GROUP BY)
- **Databricks**: Basic (notebooks, clusters)

### Quick Start

1. **Read the overview**:
   ```bash
   Start with PROJECT_OVERVIEW.md for the complete plan
   ```

2. **Begin Phase 1**:
   ```bash
   Open DATA_VALIDATION_GUIDE.md
   Create a Databricks notebook
   Run validation queries
   ```

3. **Follow sequentially**:
   - Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

## Project Structure

```
fashion-embeddings-foundation/
├── README.md                           # This file
├── PROJECT_OVERVIEW.md                 # Complete implementation plan
├── DATA_VALIDATION_GUIDE.md            # Phase 1 guide
├── VECTOR_SEARCH_MAPPING_GUIDE.md      # Phase 2 guide
├── PHASES_3_4_5_BRIEF.md              # Phases 3-5 overview
├── docs/
│   └── PRODUCTION_DATA_SNAPSHOT.md    # Technical reference
├── examples/
│   └── vector_search_examples.py      # Python code samples
├── queries/
│   └── export_taxonomy.sql            # SQL taxonomy mapping
└── schemas/
    └── product_embeddings_schema.json # Table schemas
```

## Cost Estimate

**Development** (Phases 1-4): ~$60-115
- Compute: $50-100
- Vector Search: $5-10
- Storage: <$1

**Production** (Monthly): ~$40-50
- Scheduled jobs: $38
- API serving: Shared with existing infrastructure
- Storage: <$1

## Success Criteria

By the end of this project, you will have:

✅ Validated 22K outfit embeddings
✅ 110K DeepFashion2→Product mappings
✅ Graph with 44K nodes, 100K+ edges
✅ Pre-computed recommendations with 50-65% accuracy
✅ 3 production API endpoints (<200ms latency)

## Data Sources

### Product Catalog
- **Dataset**: Fashion Product Images (Kaggle)
- **URL**: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
- **Size**: 44,424 products
- **Categories**: 143 article types across 7 master categories

### DeepFashion2
- **Purpose**: Outfit co-occurrence patterns
- **Size**: 22,000 validated outfit images
- **Use**: Graph edge construction (items that appear together)

## Support

For questions or issues:
- Review the troubleshooting sections in each guide
- Open an issue in this repository
- Contact: kevin.ippen@databricks.com

## License

MIT License - See [LICENSE](LICENSE) file for details

## Acknowledgments

- **Databricks** - Mosaic AI Vector Search, Model Serving, Unity Catalog
- **Anthropic** - Claude AI
- **OpenAI** - CLIP model
- **Kaggle** - Fashion Product Images dataset
- **DeepFashion2** - Outfit composition research data

---

**Last Updated**: 2025-12-13
**Status**: ✅ Ready to Build
**Version**: 1.0.0

🚀 **Start with [DATA_VALIDATION_GUIDE.md](DATA_VALIDATION_GUIDE.md) to begin Phase 1!**
