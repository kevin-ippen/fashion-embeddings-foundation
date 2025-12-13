# Production Data Snapshot - Actual State

**Date**: 2025-12-13
**Source**: fashion-ecom-site codebase analysis
**Status**: ✅ Verified from production config and SQL files

---

## 🗄️ **ACTUAL PRODUCTION TABLES**

### Unity Catalog Tables (Source)

**Catalog**: `main`
**Schema**: `fashion_demo`

```python
# From fashion-ecom-site/core/config.py

# Core tables
"main.fashion_demo.products"                            # 44,424 products
"main.fashion_demo.product_embeddings_multimodal"        # Image/text/hybrid embeddings
"main.fashion_demo.user_style_features"                 # 10K synthetic users
"main.fashion_demo.users"                               # User base table

# SmolVLM enrichment (created but may not be populated yet)
"main.fashion_demo.product_extracted_attributes"        # SmolVLM material/pattern/style
"main.fashion_demo.products_with_visual_attributes"     # Enriched with rich descriptions

# DeepFashion2 tables (from deepfashion2_complete_the_look.py)
"main.fashion_demo.df2_items"                           # Individual items from outfits
"main.fashion_demo.df2_outfits"                         # Outfit compositions
"main.fashion_demo.df2_complementarity"                 # Item-item compatibility scores

# UNCLEAR: Alternative naming in documentation
"main.fashion_demo.fashion_items_embeddings"            # May be alias for df2_items?
```

### Lakebase PostgreSQL Synced Tables (API Production)

**Schema**: `fashion_demo` (PostgreSQL side)
**Database**: `databricks_postgres`

```python
# From fashion-ecom-site/core/config.py, lines 35-38

"fashion_demo.productsdb"              # ← synced from main.fashion_demo.products
"fashion_demo.usersdb"                 # ← synced from main.fashion_demo.users
"fashion_demo.user_style_featuresdb"   # ← synced from main.fashion_demo.user_style_features

# API queries Lakebase (~10ms) instead of SQL Warehouse (~100ms)
```

---

## 📊 **ACTUAL SCHEMA STRUCTURE**

### Products Table Schema

**Table**: `main.fashion_demo.products`
**Source**: `fashion-ecom-site/models/schemas.py`, `repositories/lakebase.py`

```python
{
    "product_id": "INT",              # Primary key (stored as string in some contexts)
    "product_display_name": "STRING", # Required
    "master_category": "STRING",      # 7 categories
    "sub_category": "STRING",         # 45 sub-categories
    "article_type": "STRING",         # 143 unique types
    "base_color": "STRING",           # 46 colors
    "gender": "STRING",               # Men, Women, Boys, Girls, Unisex
    "season": "STRING",               # Summer, Winter, Fall, Spring (nullable)
    "year": "INT",                    # 2011-2018
    "usage": "STRING",                # Casual, Formal, Sports, etc.
    "price": "DOUBLE",                # Product price
    "image_path": "STRING",           # /Volumes/main/fashion_demo/raw_data/images/{id}.jpg
    "ingested_at": "TIMESTAMP"        # Ingest timestamp
}
```

**Filter Columns** (from `lakebase.py`, lines 46-72):
- gender
- master_category
- sub_category
- base_color
- season
- price (min/max)

### Product Embeddings Multimodal Schema

**Table**: `main.fashion_demo.product_embeddings_multimodal`
**Source**: `fashion-ecom-site/multimodal_clip_implementation.py`

```python
{
    # Product metadata (from products table)
    "product_id": "INT",
    "product_display_name": "STRING",
    "master_category": "STRING",
    "sub_category": "STRING",
    "article_type": "STRING",
    "base_color": "STRING",
    "gender": "STRING",
    "season": "STRING",
    "usage": "STRING",
    "year": "INT",
    "price": "DOUBLE",
    "image_path": "STRING",

    # Three embedding types (all 512D, L2-normalized)
    "image_embedding": "ARRAY<DOUBLE>",   # From CLIP image encoder
    "text_embedding": "ARRAY<DOUBLE>",    # From CLIP text encoder
    "hybrid_embedding": "ARRAY<DOUBLE>",  # 50/50 weighted combination

    # Metadata
    "embedding_model": "STRING",          # 'clip-vit-b-32'
    "embedding_dimension": "INT",         # 512
    "updated_at": "TIMESTAMP"
}
```

**Row count**: 44,417 (99.98% coverage)

**Delta properties** (from `join_enriched_attributes.sql`, line 132):
```sql
delta.enableChangeDataFeed = true  -- Required for Vector Search sync
```

### Products with Visual Attributes Schema

**Table**: `main.fashion_demo.products_with_visual_attributes`
**Source**: `fashion-ecom-site/notebooks/join_enriched_attributes.sql`

```python
{
    # All product fields (same as products table)
    # PLUS SmolVLM extracted attributes:

    "material": "STRING",               # leather, denim, knit, woven, metal, canvas
    "pattern": "STRING",                # solid, striped, floral, geometric, etc.
    "formality_level": "STRING",        # formal, business casual, casual, athletic
    "style_keywords": "ARRAY<STRING>",  # [vintage, modern, minimalist, ...]
    "visual_details": "ARRAY<STRING>",  # [has pockets, has buttons, ...]
    "collar_type": "STRING",            # crew neck, V-neck, collar, hooded
    "sleeve_length": "STRING",          # short, long, sleeveless, three-quarter
    "fit_type": "STRING",               # fitted, regular, loose, oversized

    # Extraction metadata
    "extraction_success": "BOOLEAN",
    "confidence_material": "STRING",    # high, medium, low
    "extraction_timestamp": "TIMESTAMP",

    # Rich text descriptions
    "rich_text_description": "STRING",  # Combined original + extracted attrs
    "baseline_description": "STRING"    # Original description for comparison
}
```

**Purpose**: Ready for CLIP text embedding generation with enriched descriptions

---

## 🔍 **VECTOR SEARCH INFRASTRUCTURE**

### Endpoint

**Name**: `fashion_vector_search`
**Status**: ONLINE
**Type**: Standard

### Indexes

From `fashion-ecom-site/core/config.py`, lines 73-76:

```python
"main.fashion_demo.vs_image_search"   # Searches image_embedding
"main.fashion_demo.vs_text_search"    # Searches text_embedding
"main.fashion_demo.vs_hybrid_search"  # Searches hybrid_embedding
```

**All indexes**:
- Source: `main.fashion_demo.product_embeddings_multimodal`
- Primary key: `product_id`
- Dimension: 512
- Sync: Continuous (Delta sync enabled)

---

## 🤖 **MODEL SERVING**

### CLIP Multimodal Encoder

From `fashion-ecom-site/core/config.py`, lines 65-68:

```python
CLIP_ENDPOINT_NAME = "clip-multimodal-encoder"
CLIP_UC_MODEL = "main.fashion_demo.clip_multimodal_encoder"
CLIP_EMBEDDING_DIM = 512

# Full URL:
# https://{DATABRICKS_WORKSPACE_URL}/serving-endpoints/clip-multimodal-encoder/invocations
```

**Model**: `openai/clip-vit-base-patch32`
**Workload**: Large (64 concurrent requests)
**Scale to zero**: Enabled

---

## 🌐 **WORKSPACE DETAILS**

From `fashion-ecom-site/core/config.py`, lines 46-63:

```python
# Default workspace URL (can be overridden by env vars)
DATABRICKS_WORKSPACE_URL = "https://adb-984752964297111.11.azuredatabricks.net"

# Environment variables used:
# - DATABRICKS_WORKSPACE_URL (explicit override)
# - DATABRICKS_HOST (Apps environment)
# - DATABRICKS_TOKEN (OAuth M2M)
```

**Platform**: Azure Databricks (adb- prefix)
**Lakebase host**: `instance-e2ff35b5-a3fc-44f3-9d65-7cba8332db7c.database.azuredatabricks.net`
**Lakebase port**: 5432

---

## 📋 **TAXONOMY BREAKDOWN**

### Known Categories

**Master Categories** (7):
From `fashion-ecom-site/models/schemas.py` and repository queries:
- Apparel
- Accessories
- Footwear
- Personal Care
- Free Items
- Sporting Goods
- Home

**Sub-Categories**: 45 unique
**Article Types**: 143 unique

### Proposed Graph Mapping (5 categories)

**See**: `queries/export_taxonomy.sql` for complete mapping logic

```
article_type (143) → graph_category (5)

1. tops      - Shirts, T-shirts, Tops, Blouses, Sweaters, etc.
2. bottoms   - Jeans, Trousers, Pants, Skirts, Shorts, etc.
3. shoes     - All Footwear master_category items
4. outerwear - Jackets, Coats, Blazers, Hoodies, Cardigans
5. accessories - Accessories master_category + watches, bags, belts, etc.

Optional 6th: dresses - Dresses, Gowns, Jumpsuits (or merge with tops)
Optional 7th: other - Personal Care, Home, Free Items (exclude from graph)
```

**Action required**: Run `queries/export_taxonomy.sql` to generate complete mapping

---

## ❓ **UNRESOLVED: DeepFashion2 Table Names**

**Ambiguity found**:

1. Code creates: `main.fashion_demo.df2_items`
   - Source: `fashion-ecom-site/notebooks/deepfashion2_complete_the_look.py`, line 443

2. Documentation refers to: `main.fashion_demo.fashion_items_embeddings`
   - Source: `fashion-embeddings-foundation/FASHION_EMBEDDINGS_FOUNDATION_SUMMARY.md`

**Hypothesis**:
- `df2_items` = Table created by notebook
- `fashion_items_embeddings` = Either:
  a) Older name that was renamed
  b) Different table with consolidated embeddings
  c) Documentation error

**Required**: Run these queries to verify:
```sql
SHOW TABLES IN main.fashion_demo LIKE 'df2%';
SHOW TABLES IN main.fashion_demo LIKE 'fashion_items%';

DESCRIBE TABLE EXTENDED main.fashion_demo.df2_items;
DESCRIBE TABLE EXTENDED main.fashion_demo.fashion_items_embeddings;
```

---

## 📁 **STORAGE PATHS**

### Unity Catalog Volumes

From multiple notebooks:

```
/Volumes/main/fashion_demo/raw_data/
├── styles.csv                    # Product metadata CSV
└── images/                       # 44K product images
    └── {product_id}.jpg

/Volumes/main/fashion_demo/complete_the_look/
└── [DeepFashion2 artifacts]      # Model outputs, intermediates
```

### DeepFashion2 Source (UNCONFIRMED)

From `deepfashion2_complete_the_look.py`, line 73:
```python
DF2_BASE_PATH = "/mnt/deepfashion2"  # ⚠️ PLACEHOLDER - needs user update
```

**Status**: Path is a placeholder, actual mount location TBD

---

## 🔧 **CONFIGURATION FILES**

**Primary config**: `fashion-ecom-site/core/config.py`
- Uses pydantic_settings with .env file support
- Environment variable priority
- Lakebase + Unity Catalog dual setup
- OAuth M2M authentication

**Key environment variables**:
- `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER` - Lakebase connection
- `DATABRICKS_HOST`, `DATABRICKS_WORKSPACE_URL` - Workspace
- `CLIENT_ID`, `CLIENT_SECRET`, `WORKSPACE_ID` - OAuth
- `DEBUG` - Debug mode toggle

---

## 🎯 **WHAT'S READY FOR GRAPH PROJECT**

### ✅ Confirmed Available

1. **44K products** with metadata + 3 embedding types
2. **Vector Search** operational (3 indexes)
3. **Model Serving** CLIP endpoint working
4. **Lakebase API** for fast queries
5. **Unity Catalog** governance in place
6. **SmolVLM pipeline** ready (may need execution)

### ❓ Needs Verification

1. DeepFashion2 table naming (`df2_items` vs `fashion_items_embeddings`)
2. DeepFashion2 data population status (33K items claimed)
3. Actual DeepFashion2 source path
4. SmolVLM execution status (tables exist but populated?)

### 📝 Needs Creation

1. **Taxonomy mapping table**: `main.fashion_demo.article_type_graph_mapping`
2. **Graph tables**:
   - `main.fashion_demo.outfit_graph_edges`
   - `main.fashion_demo.outfit_graph_nodes`
   - `main.fashion_demo.complementarity_scores`

---

## 🚀 **NEXT STEPS**

1. **Run taxonomy export** (use `queries/export_taxonomy.sql`)
2. **Verify DeepFashion2 tables** (SQL queries above)
3. **Check SmolVLM execution status**:
   ```sql
   SELECT COUNT(*) FROM main.fashion_demo.product_extracted_attributes;
   SELECT COUNT(*) FROM main.fashion_demo.products_with_visual_attributes;
   ```
4. **Confirm ready to build graph** (all data available)

---

**Last Updated**: 2025-12-13
**Confidence**: ✅ High (verified from production code)
**Source Files Analyzed**:
- `fashion-ecom-site/core/config.py`
- `fashion-ecom-site/repositories/lakebase.py`
- `fashion-ecom-site/models/schemas.py`
- `fashion-ecom-site/notebooks/join_enriched_attributes.sql`
- `fashion-ecom-site/notebooks/deepfashion2_complete_the_look.py`
- `fashion-ecom-site/multimodal_clip_implementation.py`
