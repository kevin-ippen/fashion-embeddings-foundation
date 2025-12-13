# Phase 1: Data Validation & Preparation - Implementation Guide

**Duration**: 2-3 hours  
**Dependencies**: None (start here)  
**Complexity**: Low  
**Deliverables**: 4 validated tables ready for graph construction

---

## 🎯 Phase Objectives

1. Validate 22K DeepFashion2 embeddings (quality check)
2. Validate 44K product catalog embeddings
3. Test Vector Search integration
4. Create clean working tables
5. Export complete product taxonomy

**Success Criteria**:
- ✅ All 22K DF2 embeddings are valid (512D, L2-normalized, non-zero)
- ✅ Vector Search returns results for test query
- ✅ 4 working tables created in Unity Catalog
- ✅ Product taxonomy exported (143 article types)

---

## 📊 Input Data Sources

### Table 1: DeepFashion2 CLIP Embeddings (Source)

**Table**: `main.fashion_demo.deepfashion2_clip_embeddings`  
**Count**: 22,000 items  
**Status**: ✅ Confirmed valid (0% failures)

**Schema**:
```python
{
    "item_uid": "STRING",              # Unique identifier
    "clip_embedding": "ARRAY<DOUBLE>", # 512D CLIP embedding
    "category_name": "STRING",         # DeepFashion2 category (13 types)
    # Possibly other columns (bounding boxes, style, etc.)
}
```

**Categories** (13 DeepFashion2 types):
```
short_sleeve_top, long_sleeve_top, short_sleeve_outwear, long_sleeve_outwear,
vest, sling, shorts, trousers, skirt, short_sleeve_dress, long_sleeve_dress,
vest_dress, sling_dress
```

---

### Table 2: Product Embeddings (Source)

**Table**: `main.fashion_demo.product_embeddings_multimodal`  
**Count**: 44,424 products  
**Embedding Types**: 3 per product

**Schema**:
```python
{
    "product_id": "INT",
    "product_display_name": "STRING",
    "master_category": "STRING",      # 7 categories
    "sub_category": "STRING",         # 45 sub-categories
    "article_type": "STRING",         # 143 types
    "base_color": "STRING",
    "gender": "STRING",
    "season": "STRING",
    "usage": "STRING",
    "price": "DOUBLE",
    "image_embedding": "ARRAY<DOUBLE>",   # 512D from CLIP image encoder
    "text_embedding": "ARRAY<DOUBLE>",    # 512D from CLIP text encoder
    "hybrid_embedding": "ARRAY<DOUBLE>"   # 512D weighted combo (50/50)
}
```

**Master Categories** (7):
```
Apparel, Accessories, Footwear, Personal Care, Free Items, Sporting Goods, Home
```

For graph project, use only: **Apparel, Accessories, Footwear**

---

### Table 3: Product Catalog (Source)

**Table**: `main.fashion_demo.products`  
**Count**: 44,424 products  
**Purpose**: Metadata for products

**Key columns**: Same as above minus embeddings

---

## 🔧 Implementation: Databricks Notebook

### Notebook Setup

**Name**: `01_validate_and_prepare.py`  
**Cluster**: Standard_DS3_v2 (4 cores, 14GB RAM), 0-2 workers  
**Runtime**: DBR 14.3.x

---

### Cell 1: Imports & Setup

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 1: Data Validation & Preparation
# MAGIC
# MAGIC **Objective**: Validate embeddings and create working tables
# MAGIC **Runtime**: ~5 minutes

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql.functions import col, size, expr, current_timestamp
from pyspark.sql.types import DoubleType
from collections import Counter
import matplotlib.pyplot as plt

print("✅ Imports loaded")
```

---

### Cell 2: Validate DeepFashion2 Embeddings

```python
# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Validate DeepFashion2 Embeddings

# COMMAND ----------

# Load DF2 embeddings
df2 = spark.table("main.fashion_demo.deepfashion2_clip_embeddings")

print("="*70)
print("DEEPFASHION2 DATASET")
print("="*70)
print(f"Total items: {df2.count():,}")
print(f"\nSchema:")
df2.printSchema()

# COMMAND ----------

# Validate embedding quality
embedding_stats = spark.sql("""
    SELECT 
        COUNT(*) as total_items,
        COUNT(clip_embedding) as non_null_embeddings,
        COUNT(CASE WHEN SIZE(clip_embedding) = 512 THEN 1 END) as valid_512d,
        COUNT(CASE 
            WHEN AGGREGATE(
                TRANSFORM(clip_embedding, x -> x * x),
                0.0,
                (acc, x) -> acc + x
            ) > 0.01 
        THEN 1 END) as non_zero_embeddings,
        AVG(SQRT(AGGREGATE(
            TRANSFORM(clip_embedding, x -> x * x),
            0.0,
            (acc, x) -> acc + x
        ))) as avg_l2_norm,
        MIN(SQRT(AGGREGATE(
            TRANSFORM(clip_embedding, x -> x * x),
            0.0,
            (acc, x) -> acc + x
        ))) as min_l2_norm,
        MAX(SQRT(AGGREGATE(
            TRANSFORM(clip_embedding, x -> x * x),
            0.0,
            (acc, x) -> acc + x
        ))) as max_l2_norm
    FROM main.fashion_demo.deepfashion2_clip_embeddings
""").collect()[0]

print("\n" + "="*70)
print("EMBEDDING QUALITY CHECK")
print("="*70)
print(f"Total items:          {embedding_stats.total_items:>10,}")
print(f"Non-null:             {embedding_stats.non_null_embeddings:>10,}")
print(f"Valid 512D:           {embedding_stats.valid_512d:>10,}")
print(f"Non-zero:             {embedding_stats.non_zero_embeddings:>10,}")
print(f"Avg L2 norm:          {embedding_stats.avg_l2_norm:>10.6f}")
print(f"Min L2 norm:          {embedding_stats.min_l2_norm:>10.6f}")
print(f"Max L2 norm:          {embedding_stats.max_l2_norm:>10.6f}")

# Validation checks
is_normalized = 0.99 < embedding_stats.avg_l2_norm < 1.01
all_valid = embedding_stats.non_zero_embeddings == embedding_stats.total_items

if is_normalized and all_valid:
    print(f"\n✅ ALL EMBEDDINGS VALID & NORMALIZED")
else:
    print(f"\n⚠️  ISSUES DETECTED:")
    if not is_normalized:
        print(f"   - Embeddings not normalized (mean={embedding_stats.avg_l2_norm:.4f})")
    if not all_valid:
        print(f"   - Some zero embeddings found")

# COMMAND ----------

# Category distribution
category_dist = df2.groupBy("category_name").count().orderBy(col("count").desc())

print("\nDeepFashion2 Category Distribution:")
display(category_dist)

# Save for reference
cat_pd = category_dist.toPandas()

# Visualization
plt.figure(figsize=(12, 6))
plt.bar(cat_pd['category_name'], cat_pd['count'])
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('DeepFashion2 Items by Category (22K total)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
display(plt.gcf())
plt.close()
```

**Expected Output**:
```
Total items:          22,000
Non-null:             22,000
Valid 512D:           22,000
Non-zero:             22,000
Avg L2 norm:          1.000000
✅ ALL EMBEDDINGS VALID & NORMALIZED
```

---

### Cell 3: Validate Product Embeddings

```python
# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Validate Product Catalog

# COMMAND ----------

# Load product embeddings
products_emb = spark.table("main.fashion_demo.product_embeddings_multimodal")

print("\n" + "="*70)
print("PRODUCT CATALOG")
print("="*70)
print(f"Total products: {products_emb.count():,}")

# Check embedding coverage
product_stats = products_emb.selectExpr(
    "COUNT(*) as total",
    "COUNT(image_embedding) as has_image_emb",
    "COUNT(text_embedding) as has_text_emb",
    "COUNT(hybrid_embedding) as has_hybrid_emb",
    "COUNT(CASE WHEN SIZE(image_embedding) = 512 THEN 1 END) as valid_image_512d",
    "COUNT(CASE WHEN SIZE(text_embedding) = 512 THEN 1 END) as valid_text_512d",
    "COUNT(CASE WHEN SIZE(hybrid_embedding) = 512 THEN 1 END) as valid_hybrid_512d"
).collect()[0]

print(f"\nEmbedding Coverage:")
print(f"  Total products:         {product_stats.total:>10,}")
print(f"  Image embeddings:       {product_stats.has_image_emb:>10,} ({product_stats.valid_image_512d:,} valid 512D)")
print(f"  Text embeddings:        {product_stats.has_text_emb:>10,} ({product_stats.valid_text_512d:,} valid 512D)")
print(f"  Hybrid embeddings:      {product_stats.has_hybrid_emb:>10,} ({product_stats.valid_hybrid_512d:,} valid 512D)")

# Sample
print(f"\nSample products:")
display(products_emb.select("product_id", "product_display_name", "article_type", "master_category", "price").limit(10))
```

**Expected Output**:
```
Total products:         44,424
Image embeddings:       44,417 (44,417 valid 512D)
Text embeddings:        44,424 (44,424 valid 512D)
Hybrid embeddings:      44,417 (44,417 valid 512D)
```

---

### Cell 4: Test Vector Search

```python
# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test Vector Search Integration

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

try:
    vsc = VectorSearchClient()
    
    # Check endpoint
    endpoint = vsc.get_endpoint("fashion_vector_search")
    print("✅ Vector Search endpoint: fashion_vector_search")
    
    # Check image index (we'll use this for DF2 mapping)
    image_index = vsc.get_index(
        endpoint_name="fashion_vector_search",
        index_name="main.fashion_demo.vs_image_search"
    )
    print("✅ Image search index available")
    
    # Test query with DF2 embedding
    test_df2_item = df2.limit(1).collect()[0]
    
    print(f"\nTest Query:")
    print(f"  DF2 Item: {test_df2_item.item_uid}")
    print(f"  Category: {test_df2_item.category_name}")
    
    # Search for similar products
    results = image_index.similarity_search(
        query_vector=test_df2_item.clip_embedding,
        num_results=5
    )
    
    print(f"\n  Top-5 similar products:")
    for i, result in enumerate(results['result']['data_array'], 1):
        print(f"    {i}. Product {result['product_id']}: {result['product_display_name']}")
        print(f"       Similarity: {result['score']:.4f}, Type: {result['article_type']}")
    
    print(f"\n✅ Vector Search working correctly")
    
except Exception as e:
    print(f"❌ Vector Search error: {e}")
    print(f"   Check that endpoint 'fashion_vector_search' is online")
```

**Expected Output**:
```
✅ Vector Search endpoint: fashion_vector_search
✅ Image search index available

Test Query:
  DF2 Item: 12345
  Category: long_sleeve_top

  Top-5 similar products:
    1. Product 789: Blue Denim Shirt
       Similarity: 0.8523, Type: Shirts
    2. Product 456: Cotton T-Shirt
       Similarity: 0.8201, Type: Tshirts
    ...

✅ Vector Search working correctly
```

---

### Cell 5: Create Working Tables

```python
# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Working Tables

# COMMAND ----------

# Table 1: DF2 Working Set
spark.sql("""
    CREATE OR REPLACE TABLE main.fashion_demo.df2_working_set AS
    SELECT 
        *,
        SQRT(AGGREGATE(TRANSFORM(clip_embedding, x -> x*x), 0.0, (acc,x) -> acc+x)) as l2_norm,
        current_timestamp() as created_at
    FROM main.fashion_demo.deepfashion2_clip_embeddings
    WHERE SIZE(clip_embedding) = 512
      AND AGGREGATE(TRANSFORM(clip_embedding, x -> x*x), 0.0, (acc,x) -> acc+x) > 0.01
""")

df2_count = spark.table("main.fashion_demo.df2_working_set").count()
print(f"✅ Created: main.fashion_demo.df2_working_set")
print(f"   Rows: {df2_count:,}")

# COMMAND ----------

# Table 2: Products Working Set (filtered to fashion categories)
spark.sql("""
    CREATE OR REPLACE TABLE main.fashion_demo.products_working_set AS
    SELECT 
        p.product_id,
        p.product_display_name,
        p.master_category,
        p.sub_category,
        p.article_type,
        p.base_color,
        p.gender,
        p.season,
        p.usage,
        p.price,
        p.image_path,
        e.image_embedding,
        e.text_embedding,
        e.hybrid_embedding,
        current_timestamp() as created_at
    FROM main.fashion_demo.products p
    JOIN main.fashion_demo.product_embeddings_multimodal e
        ON p.product_id = e.product_id
    WHERE p.master_category IN ('Apparel', 'Accessories', 'Footwear')
        AND e.image_embedding IS NOT NULL
""")

prod_count = spark.table("main.fashion_demo.products_working_set").count()
print(f"✅ Created: main.fashion_demo.products_working_set")
print(f"   Rows: {prod_count:,}")
print(f"   Filtered from 44K → {prod_count:,} (fashion categories only)")
```

**Expected Output**:
```
✅ Created: main.fashion_demo.df2_working_set
   Rows: 22,000

✅ Created: main.fashion_demo.products_working_set
   Rows: ~42,000-43,000
   Filtered from 44K → 42,567 (fashion categories only)
```

---

### Cell 6: Export Product Taxonomy

```python
# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Export Product Taxonomy

# COMMAND ----------

# Get complete article type taxonomy
taxonomy = spark.sql("""
    SELECT 
        article_type,
        master_category,
        sub_category,
        COUNT(*) as product_count,
        MIN(price) as min_price,
        MAX(price) as max_price,
        AVG(price) as avg_price
    FROM main.fashion_demo.products_working_set
    GROUP BY article_type, master_category, sub_category
    ORDER BY master_category, product_count DESC
""")

print(f"\nProduct Taxonomy:")
print(f"  Unique article types: {taxonomy.count()}")
print(f"  Master categories: {taxonomy.select('master_category').distinct().count()}")
print(f"  Sub-categories: {taxonomy.select('sub_category').distinct().count()}")

display(taxonomy)

# Save to table
taxonomy.write.mode("overwrite").saveAsTable("main.fashion_demo.product_taxonomy")
print(f"\n✅ Saved: main.fashion_demo.product_taxonomy")

# Export to pandas for manual inspection
taxonomy_pd = taxonomy.toPandas()
print(f"\nTop 20 article types by count:")
print(taxonomy_pd.head(20)[['article_type', 'master_category', 'product_count']])
```

**Expected Output**:
```
Product Taxonomy:
  Unique article types: ~143
  Master categories: 3 (Apparel, Accessories, Footwear)
  Sub-categories: ~40-45

Top 20 article types by count:
   article_type         master_category  product_count
0  Tshirts              Apparel          5,234
1  Shirts               Apparel          4,892
2  Casual Shoes         Footwear         3,567
...
```

---

### Cell 7: Summary & Validation

```python
# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary & Readiness Check

# COMMAND ----------

# Compile summary
summary = {
    "df2_items": df2_count,
    "products": prod_count,
    "df2_categories": df2.select("category_name").distinct().count(),
    "product_article_types": taxonomy.count(),
    "vector_search_ready": True,  # Set based on Cell 4 result
    "embeddings_valid": embedding_stats.non_zero_embeddings == embedding_stats.total_items,
    "embeddings_normalized": 0.99 < embedding_stats.avg_l2_norm < 1.01,
}

print("\n" + "="*70)
print("PHASE 1 VALIDATION SUMMARY")
print("="*70)
print(f"DeepFashion2 items:        {summary['df2_items']:>10,}")
print(f"Product catalog:           {summary['products']:>10,}")
print(f"DF2 categories:            {summary['df2_categories']:>10}")
print(f"Product article types:     {summary['product_article_types']:>10}")
print(f"Vector Search ready:       {'✅' if summary['vector_search_ready'] else '❌'}")
print(f"Embeddings valid:          {'✅' if summary['embeddings_valid'] else '❌'}")
print(f"Embeddings normalized:     {'✅' if summary['embeddings_normalized'] else '❌'}")
print("="*70)

# Check readiness
all_ready = all([
    summary['df2_items'] >= 20000,
    summary['products'] >= 40000,
    summary['vector_search_ready'],
    summary['embeddings_valid'],
    summary['embeddings_normalized']
])

if all_ready:
    print("\n🎉 PHASE 1 COMPLETE - ALL SYSTEMS GO!")
    print("\nDeliverables created:")
    print("  ✅ main.fashion_demo.df2_working_set")
    print("  ✅ main.fashion_demo.products_working_set")
    print("  ✅ main.fashion_demo.product_taxonomy")
    print("\nNext step: Phase 2 - Vector Search Mapping")
    print("  → Use VECTOR_SEARCH_MAPPING_GUIDE.md")
else:
    print("\n⚠️  ISSUES DETECTED - Review above before proceeding")

# Save summary
summary_df = spark.createDataFrame([{
    "phase": "1_validation",
    "validation_date": pd.Timestamp.now(),
    **summary
}])

summary_df.write.mode("overwrite").saveAsTable(
    "main.fashion_demo.phase1_validation_summary"
)

print(f"\n✅ Validation summary saved: main.fashion_demo.phase1_validation_summary")
```

**Expected Output**:
```
PHASE 1 VALIDATION SUMMARY
======================================================================
DeepFashion2 items:            22,000
Product catalog:               42,567
DF2 categories:                    13
Product article types:            143
Vector Search ready:               ✅
Embeddings valid:                  ✅
Embeddings normalized:             ✅
======================================================================

🎉 PHASE 1 COMPLETE - ALL SYSTEMS GO!

Deliverables created:
  ✅ main.fashion_demo.df2_working_set
  ✅ main.fashion_demo.products_working_set
  ✅ main.fashion_demo.product_taxonomy

Next step: Phase 2 - Vector Search Mapping
  → Use VECTOR_SEARCH_MAPPING_GUIDE.md

✅ Validation summary saved: main.fashion_demo.phase1_validation_summary
```

---

## ✅ Phase 1 Deliverables

### Tables Created

1. **`main.fashion_demo.df2_working_set`**
   - 22,000 DeepFashion2 items with valid embeddings
   - Added: `l2_norm`, `created_at`

2. **`main.fashion_demo.products_working_set`**
   - ~42K products (Apparel, Accessories, Footwear only)
   - Combined product metadata + 3 embedding types

3. **`main.fashion_demo.product_taxonomy`**
   - 143 article types with counts, price ranges
   - Master category → sub-category → article type hierarchy

4. **`main.fashion_demo.phase1_validation_summary`**
   - Validation metrics and readiness flags

---

## 🎯 Success Validation

**Before moving to Phase 2, verify**:

```sql
-- Check table existence
SHOW TABLES IN main.fashion_demo LIKE '%working_set%';

-- Check row counts
SELECT COUNT(*) FROM main.fashion_demo.df2_working_set;        -- Should be 22,000
SELECT COUNT(*) FROM main.fashion_demo.products_working_set;   -- Should be ~42K

-- Check taxonomy
SELECT COUNT(DISTINCT article_type) FROM main.fashion_demo.product_taxonomy;  -- ~143
```

All queries should return expected counts. ✅

---

## 🚨 Troubleshooting

### Issue: Vector Search test fails

**Error**: `Endpoint not found: fashion_vector_search`

**Solution**:
- Check endpoint name in Databricks Compute → Vector Search
- Verify endpoint is ONLINE (not stopped)
- Try alternative indexes: `vs_text_search` or `vs_hybrid_search`

---

### Issue: Zero embeddings found

**Error**: `non_zero_embeddings < total_items`

**Solution**:
- You're using wrong table - switch to `deepfashion2_clip_embeddings`
- Or run additional filter in working table creation

---

### Issue: Missing columns

**Error**: `Column 'clip_embedding' does not exist`

**Solution**:
- Check actual column name: `DESCRIBE TABLE main.fashion_demo.deepfashion2_clip_embeddings`
- Update all references to correct column name
- Common alternatives: `image_embedding`, `embedding`

---

### Issue: Out of memory

**Error**: `OutOfMemoryError` during embedding stats calculation

**Solution**:
- Use larger cluster (Standard_DS4_v2 with 28GB RAM)
- Add workers (2-4 workers)
- Sample data for validation: `.sample(0.1)` for 10%

---

## 📋 Next Steps

Once Phase 1 completes successfully:

1. ✅ All 4 tables created
2. ✅ All validation checks pass
3. ✅ Vector Search tested

**→ Proceed to Phase 2**: `VECTOR_SEARCH_MAPPING_GUIDE.md`

**What Phase 2 does**:
- Maps 22K DF2 items → products using Vector Search
- Creates ~110K mappings (22K × 5 top matches)
- Quality validation & manual review

---

**Phase 1 Complete!** ✅ Ready for Phase 2.
