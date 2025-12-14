# Vector Search Setup Required for Phase 2

## Current Status

✅ **Phase 1 Complete** - Data validation successful
- 22,000 DeepFashion2 embeddings validated (all 512D)
- 44,417 product embeddings validated (100% coverage)
- Working tables created
- Taxonomy mapping complete (119 article types)

⏸️ **Phase 2 Blocked** - Waiting for Vector Search endpoint

## What Phase 2 Needs

Phase 2 (Vector Search Mapping) requires:

1. **Vector Search Endpoint** with access to product embeddings
2. **Vector Search Index** on the products table with hybrid embeddings

## Required Vector Search Index

### Index Configuration

**Table**: `main.fashion_demo.products_working_set` (41,888 products)

**Index Name**: `main.fashion_demo.vs_product_hybrid_search` (or similar)

**Embedding Column**: `hybrid_embedding` (512D CLIP embeddings)

**Endpoint**: Any available Vector Search endpoint (e.g., `one-env-shared-endpoint-1`)

### Columns to Include in Index

```python
[
    "product_id",           # Primary key
    "product_display_name", # Product name
    "article_type",         # Article type (e.g., "Tshirts")
    "master_category",      # Category (e.g., "Apparel")
    "price",               # Price
    "base_color",          # Color (optional)
    "gender"               # Gender (optional)
]
```

## SQL to Create Vector Search Index

```sql
-- Example SQL (adjust endpoint name as needed)
CREATE VECTOR SEARCH INDEX IF NOT EXISTS main.fashion_demo.vs_product_hybrid_search
ON main.fashion_demo.products_working_set (
    hybrid_embedding
)
WITH (
    endpoint_name = 'one-env-shared-endpoint-1',  -- Adjust to your endpoint
    primary_key = 'product_id',
    embedding_dimension = 512,
    embedding_model = 'clip-vit-b-32'
)
SELECT
    product_id,
    product_display_name,
    article_type,
    master_category,
    price,
    base_color,
    gender,
    hybrid_embedding
FROM main.fashion_demo.products_working_set;
```

## Alternative: Use Databricks UI

1. Navigate to **Compute** → **Vector Search**
2. Click **Create Index**
3. Configure:
   - **Source Table**: `main.fashion_demo.products_working_set`
   - **Endpoint**: Select any available endpoint
   - **Primary Key**: `product_id`
   - **Embedding Column**: `hybrid_embedding`
   - **Embedding Dimension**: 512
4. Select columns to include (see list above)
5. Click **Create**

## What Happens Once Setup

Once the Vector Search index is ready, Phase 2 will:

1. **Load 22,000 DeepFashion2 embeddings** from working set
2. **Query Vector Search** for top-5 similar products for each DF2 item
3. **Create ~110,000 mappings** (22K × 5)
4. **Save to Delta table**: `main.fashion_demo.df2_to_product_mappings`
5. **Perform quality analysis**:
   - Similarity score distribution
   - Coverage statistics
   - Most frequently mapped products

**Expected Runtime**: 15-30 minutes (depending on batch size and rate limits)

## Phase 2 Script Ready

The Phase 2 script is already created and ready to run:
- **Script**: `run_phase2_mapping.py`
- **Configuration**: TOP_K = 5, BATCH_SIZE = 100
- **Rate limiting**: 0.5s between batches

## How to Resume Phase 2

Once Vector Search is set up:

```bash
cd /Users/kevin.ippen/projects/fashion-embeddings-foundation

# Update the script with correct endpoint/index names if needed
# Then run:
python3 run_phase2_mapping.py
```

The script will automatically:
- Connect to the Vector Search index
- Process all 22K items in batches
- Save results to Delta
- Generate quality report

## Current Data Summary

**Tables Ready**:
- ✅ `main.fashion_demo.df2_working_set` (22,000 rows)
- ✅ `main.fashion_demo.products_working_set` (41,888 rows)
- ✅ `main.fashion_demo.product_taxonomy` (119 mappings)
- ✅ `main.fashion_demo.phase1_validation_summary` (summary)

**Next Table to Create**:
- ⏸️ `main.fashion_demo.df2_to_product_mappings` (~110K rows) - Phase 2 output

## Contact

Once Vector Search is configured, ping me and I'll resume Phase 2 immediately! 🚀

---

**Last Updated**: 2025-12-13
**Status**: Waiting for Vector Search setup
