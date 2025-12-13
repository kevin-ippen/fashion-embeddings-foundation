# Phase 2: Vector Search Mapping - Implementation Guide

**Duration**: 3-4 hours  
**Dependencies**: Phase 1 complete  
**Complexity**: Medium  
**Deliverables**: DF2→Product mapping table (~110K mappings)

---

## 🎯 Phase Objectives

Map 22,000 DeepFashion2 outfit items to product catalog using Vector Search:
1. For each DF2 item, find top-5 most similar products
2. Use cosine similarity on image embeddings
3. Apply category constraints (optional)
4. Store mappings in Delta table
5. Quality validation & manual review

**Success Criteria**:
- ✅ ~110K mappings created (22K items × 5 products)
- ✅ Average similarity score >0.6
- ✅ Manual review: 70%+ positive ratings
- ✅ Category alignment: 80%+ items map to logical categories

---

## 📊 Input Data

### From Phase 1:

**Table**: `main.fashion_demo.df2_working_set`  
**Count**: 22,000 items  
**Columns needed**:
- `item_uid` - Unique ID
- `clip_embedding` - 512D CLIP embedding (L2-normalized)
- `category_name` - DeepFashion2 category (13 types)

**Table**: `main.fashion_demo.products_working_set`  
**Count**: ~42,000 products  
**Columns needed**:
- `product_id` - Product ID
- `image_embedding` - 512D CLIP image embedding
- `article_type` - Product type
- `master_category` - Category

---

## 🔧 Mapping Strategy

### Top-5 Similarity Matching

**Why top-5?**
- More graph density (22K × 5 = 110K edges)
- Captures variety in product catalog
- Balances precision vs coverage
- Typical outfit has 3-5 items

**Alternative**: Top-3 (66K mappings, more precise but sparser)

### Vector Search Configuration

**Index**: `main.fashion_demo.vs_image_search`  
**Why image embeddings?**
- DF2 items are outfit photos → visual similarity
- Image encoder captures appearance
- Consistent with product photo embeddings

**Query parameters**:
- `num_results`: 5
- `filters`: Optional category constraints
- `query_type`: "ANN" (approximate nearest neighbors)

---

## 🔧 Implementation: Databricks Notebook

### Notebook Setup

**Name**: `02_map_df2_to_products.py`  
**Cluster**: Standard_DS3_v2, 0-2 workers  
**Runtime**: DBR 14.3.x  
**Duration**: ~20-30 minutes

---

### Cell 1: Setup & Load Data

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2: Vector Search Mapping
# MAGIC
# MAGIC **Objective**: Map 22K DF2 items → products via Vector Search
# MAGIC **Strategy**: Top-5 similarity matching
# MAGIC **Runtime**: ~20-30 minutes

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from pyspark.sql.functions import col, explode, struct, current_timestamp
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import time

# COMMAND ----------

# Load DF2 items
df2_items = spark.table("main.fashion_demo.df2_working_set")

print(f"DeepFashion2 items to map: {df2_items.count():,}")

# Load product metadata for reference
products = spark.table("main.fashion_demo.products_working_set")

print(f"Target product catalog: {products.count():,}")

# COMMAND ----------

# Initialize Vector Search client
vsc = VectorSearchClient()

# Get image search index
try:
    image_index = vsc.get_index(
        endpoint_name="fashion_vector_search",
        index_name="main.fashion_demo.vs_image_search"
    )
    print("✅ Vector Search index loaded: vs_image_search")
except Exception as e:
    print(f"❌ Error loading Vector Search index: {e}")
    raise
```

---

### Cell 2: Define Mapping Function

```python
# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Define Mapping Function

# COMMAND ----------

def map_df2_item_to_products(item_uid, embedding, category_name, index, num_results=5):
    """
    Map one DF2 item to top-N similar products
    
    Args:
        item_uid: DF2 item identifier
        embedding: 512D CLIP embedding (list)
        category_name: DF2 category (for optional filtering)
        index: Vector Search index object
        num_results: Number of top matches to return
    
    Returns:
        List of dicts with mapping info
    """
    try:
        # Query Vector Search
        results = index.similarity_search(
            query_vector=embedding,
            num_results=num_results
            # Optional: Add category filter for better alignment
            # filters={"master_category": "Apparel"}
        )
        
        # Extract results
        mappings = []
        for i, result in enumerate(results['result']['data_array']):
            mappings.append({
                'df2_item_uid': item_uid,
                'df2_category': category_name,
                'product_id': result['product_id'],
                'product_name': result['product_display_name'],
                'product_article_type': result['article_type'],
                'product_master_category': result['master_category'],
                'similarity_score': float(result['score']),
                'rank': i + 1  # 1-5
            })
        
        return mappings
    
    except Exception as e:
        print(f"Error mapping item {item_uid}: {e}")
        return []

# COMMAND ----------

# Test on one item
test_item = df2_items.limit(1).collect()[0]

test_mappings = map_df2_item_to_products(
    item_uid=test_item.item_uid,
    embedding=test_item.clip_embedding,
    category_name=test_item.category_name,
    index=image_index,
    num_results=5
)

print(f"Test mapping for DF2 item: {test_item.item_uid}")
print(f"  Category: {test_item.category_name}")
print(f"\n  Top-5 matches:")
for mapping in test_mappings:
    print(f"    {mapping['rank']}. {mapping['product_name']}")
    print(f"       Similarity: {mapping['similarity_score']:.4f}")
    print(f"       Type: {mapping['product_article_type']}")

if len(test_mappings) == 5:
    print(f"\n✅ Mapping function working correctly")
else:
    print(f"\n⚠️  Expected 5 mappings, got {len(test_mappings)}")
```

**Expected Output**:
```
Test mapping for DF2 item: 000001
  Category: long_sleeve_top

  Top-5 matches:
    1. Blue Denim Shirt
       Similarity: 0.8523
       Type: Shirts
    2. Cotton Casual Shirt
       Similarity: 0.8201
       Type: Shirts
    ...

✅ Mapping function working correctly
```

---

### Cell 3: Batch Mapping (Option 1: Simple Loop)

```python
# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Map All DF2 Items to Products

# COMMAND ----------

# Collect DF2 items (small dataset, can fit in memory)
df2_items_list = df2_items.select(
    "item_uid", 
    "clip_embedding", 
    "category_name"
).collect()

print(f"Mapping {len(df2_items_list):,} DF2 items...")
print(f"Expected mappings: {len(df2_items_list) * 5:,}")
print(f"Estimated time: {len(df2_items_list) * 0.05:.1f} seconds (~{len(df2_items_list) * 0.05 / 60:.1f} minutes)")

# COMMAND ----------

# Map all items with progress bar
all_mappings = []

for item in tqdm(df2_items_list, desc="Mapping DF2→Products"):
    mappings = map_df2_item_to_products(
        item_uid=item.item_uid,
        embedding=item.clip_embedding,
        category_name=item.category_name,
        index=image_index,
        num_results=5
    )
    all_mappings.extend(mappings)
    
    # Optional: Rate limiting to avoid quota issues
    # time.sleep(0.01)  # 10ms delay

print(f"\n✅ Mapping complete!")
print(f"   Total mappings created: {len(all_mappings):,}")
```

**Expected Output**:
```
Mapping 22,000 DF2 items...
Expected mappings: 110,000
Estimated time: 1100.0 seconds (~18.3 minutes)

Mapping DF2→Products: 100%|██████████| 22000/22000 [18:23<00:00, 19.94it/s]

✅ Mapping complete!
   Total mappings created: 110,000
```

---

### Cell 4: Batch Mapping (Option 2: Parallel with Batching)

**Alternative for faster processing**:

```python
# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Map All DF2 Items (Parallel Batches)

# COMMAND ----------

from concurrent.futures import ThreadPoolExecutor, as_completed

def map_batch(batch_items, index, batch_num):
    """Map a batch of items"""
    batch_mappings = []
    for item in batch_items:
        mappings = map_df2_item_to_products(
            item_uid=item.item_uid,
            embedding=item.clip_embedding,
            category_name=item.category_name,
            index=index,
            num_results=5
        )
        batch_mappings.extend(mappings)
    
    print(f"Batch {batch_num} complete: {len(batch_mappings)} mappings")
    return batch_mappings

# COMMAND ----------

# Split into batches
batch_size = 1000  # 1K items per batch
batches = [
    df2_items_list[i:i+batch_size] 
    for i in range(0, len(df2_items_list), batch_size)
]

print(f"Processing {len(batches)} batches of ~{batch_size} items each")

# COMMAND ----------

# Process batches in parallel
all_mappings = []

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(map_batch, batch, image_index, i)
        for i, batch in enumerate(batches)
    ]
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="Batches"):
        batch_mappings = future.result()
        all_mappings.extend(batch_mappings)

print(f"\n✅ Parallel mapping complete!")
print(f"   Total mappings: {len(all_mappings):,}")
```

**Note**: Use Option 1 (simple loop) for reliability, Option 2 (parallel) for speed if no rate limits.

---

### Cell 5: Save Mappings to Delta

```python
# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Save Mappings to Delta Table

# COMMAND ----------

# Convert to Spark DataFrame
mappings_df = spark.createDataFrame(all_mappings)

# Add metadata
mappings_df = mappings_df.withColumn("created_at", current_timestamp())

print(f"Mappings DataFrame:")
print(f"  Rows: {mappings_df.count():,}")
mappings_df.printSchema()

display(mappings_df.limit(10))

# COMMAND ----------

# Save to Delta table
mappings_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("main.fashion_demo.df2_to_product_mappings")

print(f"✅ Saved: main.fashion_demo.df2_to_product_mappings")
print(f"   Rows: {spark.table('main.fashion_demo.df2_to_product_mappings').count():,}")
```

**Expected Output**:
```
Mappings DataFrame:
  Rows: 110,000

✅ Saved: main.fashion_demo.df2_to_product_mappings
   Rows: 110,000
```

---

### Cell 6: Quality Analysis

```python
# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Mapping Quality Analysis

# COMMAND ----------

# Load saved mappings
mappings = spark.table("main.fashion_demo.df2_to_product_mappings")

# Overall statistics
stats = mappings.selectExpr(
    "COUNT(DISTINCT df2_item_uid) as unique_df2_items",
    "COUNT(DISTINCT product_id) as unique_products_mapped",
    "COUNT(*) as total_mappings",
    "AVG(similarity_score) as avg_similarity",
    "MIN(similarity_score) as min_similarity",
    "MAX(similarity_score) as max_similarity"
).collect()[0]

print("="*70)
print("MAPPING STATISTICS")
print("="*70)
print(f"DF2 items mapped:          {stats.unique_df2_items:>10,}")
print(f"Unique products mapped to: {stats.unique_products_mapped:>10,}")
print(f"Total mappings:            {stats.total_mappings:>10,}")
print(f"Avg similarity score:      {stats.avg_similarity:>10.4f}")
print(f"Min similarity score:      {stats.min_similarity:>10.4f}")
print(f"Max similarity score:      {stats.max_similarity:>10.4f}")
print("="*70)

# COMMAND ----------

# Similarity distribution
similarity_dist = spark.sql("""
    SELECT 
        CASE 
            WHEN similarity_score >= 0.9 THEN '0.90-1.00 (Excellent)'
            WHEN similarity_score >= 0.8 THEN '0.80-0.90 (Very Good)'
            WHEN similarity_score >= 0.7 THEN '0.70-0.80 (Good)'
            WHEN similarity_score >= 0.6 THEN '0.60-0.70 (Acceptable)'
            WHEN similarity_score >= 0.5 THEN '0.50-0.60 (Fair)'
            ELSE '< 0.50 (Poor)'
        END as similarity_range,
        COUNT(*) as count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
    FROM main.fashion_demo.df2_to_product_mappings
    GROUP BY 1
    ORDER BY MIN(similarity_score) DESC
""")

print("\nSimilarity Score Distribution:")
display(similarity_dist)

# COMMAND ----------

# Category alignment analysis
category_alignment = spark.sql("""
    SELECT 
        df2_category,
        product_master_category,
        product_article_type,
        COUNT(*) as mapping_count,
        AVG(similarity_score) as avg_similarity
    FROM main.fashion_demo.df2_to_product_mappings
    GROUP BY df2_category, product_master_category, product_article_type
    ORDER BY df2_category, mapping_count DESC
""")

print("\nCategory Alignment (Top mappings by DF2 category):")
display(category_alignment)
```

**Expected Output**:
```
MAPPING STATISTICS
======================================================================
DF2 items mapped:              22,000
Unique products mapped to:     ~35,000-40,000
Total mappings:               110,000
Avg similarity score:            0.7234
Min similarity score:            0.4523
Max similarity score:            0.9876
======================================================================

Similarity Score Distribution:
  0.80-0.90 (Very Good):  35,234 (32.0%)
  0.70-0.80 (Good):       48,567 (44.2%)
  0.60-0.70 (Acceptable): 21,345 (19.4%)
  ...
```

---

### Cell 7: Manual Quality Review

```python
# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Manual Quality Review (Sample)

# COMMAND ----------

# Sample 50 random mappings for manual review
sample_for_review = mappings \
    .filter(col("rank") == 1) \
    .sample(0.002) \
    .limit(50) \
    .select(
        "df2_item_uid",
        "df2_category",
        "product_id",
        "product_name",
        "product_article_type",
        "similarity_score"
    )

print("Sample mappings for manual review (Rank 1 only):")
display(sample_for_review)

# COMMAND ----------

# Create review template
review_template = sample_for_review.toPandas()
review_template['manual_rating'] = None  # Add column for rating (1-5)
review_template['notes'] = None          # Add column for notes

# Save for review
review_template.to_csv("/dbfs/tmp/mapping_quality_review.csv", index=False)

print(f"✅ Review template saved: /dbfs/tmp/mapping_quality_review.csv")
print(f"   Please review 50 samples and rate 1-5:")
print(f"     5 - Excellent match")
print(f"     4 - Very good match")
print(f"     3 - Acceptable match")
print(f"     2 - Poor match")
print(f"     1 - Wrong match")
```

**Manual Review Process**:
1. Download `/dbfs/tmp/mapping_quality_review.csv`
2. For each row, rate the DF2→Product mapping quality
3. Upload back and analyze (optional)

---

### Cell 8: Category Mapping Rules (Optional Enhancement)

```python
# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Create Category Mapping Rules

# COMMAND ----------

# Define logical DF2→Product category mappings
category_mapping_rules = {
    # DF2 tops → Product tops
    'short_sleeve_top': ['Shirts', 'Tshirts', 'Tops'],
    'long_sleeve_top': ['Shirts', 'Sweaters', 'Sweatshirts'],
    'vest': ['Tops', 'Innerwear Vests'],
    'sling': ['Tops', 'Camisoles'],
    
    # DF2 outerwear → Product outerwear
    'short_sleeve_outwear': ['Jackets', 'Shirts'],
    'long_sleeve_outwear': ['Jackets', 'Coats', 'Blazers'],
    
    # DF2 bottoms → Product bottoms
    'shorts': ['Shorts'],
    'trousers': ['Trousers', 'Jeans', 'Track Pants'],
    'skirt': ['Skirts'],
    
    # DF2 dresses → Product dresses
    'short_sleeve_dress': ['Dresses'],
    'long_sleeve_dress': ['Dresses'],
    'vest_dress': ['Dresses'],
    'sling_dress': ['Dresses']
}

# Convert to table for reference
category_rules_data = [
    {'df2_category': k, 'product_article_types': v}
    for k, v in category_mapping_rules.items()
]

category_rules_df = spark.createDataFrame(category_rules_data)
category_rules_df.write.mode("overwrite").saveAsTable(
    "main.fashion_demo.df2_category_mapping_rules"
)

print(f"✅ Category mapping rules saved")
print(f"   Use these for validation and filtering")

# COMMAND ----------

# Validate mappings against rules
validation = spark.sql("""
    WITH mapping_counts AS (
        SELECT 
            m.df2_category,
            m.product_article_type,
            COUNT(*) as count
        FROM main.fashion_demo.df2_to_product_mappings m
        GROUP BY 1, 2
    )
    SELECT 
        mc.df2_category,
        mc.product_article_type,
        mc.count,
        CASE 
            WHEN r.df2_category IS NOT NULL THEN 'Expected'
            ELSE 'Unexpected'
        END as mapping_status
    FROM mapping_counts mc
    LEFT JOIN main.fashion_demo.df2_category_mapping_rules r
        ON mc.df2_category = r.df2_category
        AND ARRAY_CONTAINS(r.product_article_types, mc.product_article_type)
    ORDER BY mc.df2_category, mc.count DESC
""")

print("\nMapping Validation (Expected vs Unexpected):")
display(validation)
```

---

### Cell 9: Summary & Next Steps

```python
# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Phase 2 Summary

# COMMAND ----------

# Compute final summary
final_summary = {
    "phase": "2_mapping",
    "df2_items_mapped": stats.unique_df2_items,
    "products_mapped_to": stats.unique_products_mapped,
    "total_mappings": stats.total_mappings,
    "avg_similarity": float(stats.avg_similarity),
    "min_similarity": float(stats.min_similarity),
    "max_similarity": float(stats.max_similarity),
    "mappings_above_0_7": mappings.filter(col("similarity_score") > 0.7).count(),
    "mappings_above_0_6": mappings.filter(col("similarity_score") > 0.6).count(),
    "created_at": pd.Timestamp.now()
}

print("="*70)
print("PHASE 2 SUMMARY")
print("="*70)
print(f"DF2 items mapped:              {final_summary['df2_items_mapped']:>10,}")
print(f"Products mapped to:            {final_summary['products_mapped_to']:>10,}")
print(f"Total mappings:                {final_summary['total_mappings']:>10,}")
print(f"Average similarity:            {final_summary['avg_similarity']:>10.4f}")
print(f"Mappings with sim > 0.7:       {final_summary['mappings_above_0_7']:>10,} ({final_summary['mappings_above_0_7']/final_summary['total_mappings']*100:.1f}%)")
print(f"Mappings with sim > 0.6:       {final_summary['mappings_above_0_6']:>10,} ({final_summary['mappings_above_0_6']/final_summary['total_mappings']*100:.1f}%)")
print("="*70)

# Quality assessment
quality_good = (
    final_summary['avg_similarity'] > 0.65 and
    final_summary['mappings_above_0_7'] > final_summary['total_mappings'] * 0.4
)

if quality_good:
    print("\n✅ MAPPING QUALITY: GOOD")
    print("   Average similarity >0.65 and 40%+ mappings >0.7")
else:
    print("\n⚠️  MAPPING QUALITY: REVIEW NEEDED")
    print(f"   Avg similarity: {final_summary['avg_similarity']:.4f}")
    print(f"   Consider increasing similarity threshold")

# Save summary
summary_df = spark.createDataFrame([final_summary])
summary_df.write.mode("overwrite").saveAsTable(
    "main.fashion_demo.phase2_mapping_summary"
)

print(f"\n✅ Summary saved: main.fashion_demo.phase2_mapping_summary")

# COMMAND ----------

print("\n" + "="*70)
print("PHASE 2 COMPLETE!")
print("="*70)
print("\nDeliverables created:")
print("  ✅ main.fashion_demo.df2_to_product_mappings (110K rows)")
print("  ✅ main.fashion_demo.df2_category_mapping_rules")
print("  ✅ main.fashion_demo.phase2_mapping_summary")
print("\nNext step: Phase 3 - Graph Construction")
print("  → Use GRAPH_CONSTRUCTION_GUIDE.md")
print("="*70)
```

---

## ✅ Phase 2 Deliverables

### Tables Created

1. **`main.fashion_demo.df2_to_product_mappings`** (~110K rows)
   ```
   df2_item_uid, df2_category, product_id, product_name,
   product_article_type, similarity_score, rank
   ```

2. **`main.fashion_demo.df2_category_mapping_rules`**
   - Logical category alignments for validation

3. **`main.fashion_demo.phase2_mapping_summary`**
   - Quality metrics and statistics

---

## 🎯 Success Validation

```sql
-- Check mappings table
SELECT COUNT(*) FROM main.fashion_demo.df2_to_product_mappings;  -- Should be ~110K

-- Check quality
SELECT 
    AVG(similarity_score) as avg_sim,
    COUNT(CASE WHEN similarity_score > 0.7 THEN 1 END) * 100.0 / COUNT(*) as pct_above_0_7
FROM main.fashion_demo.df2_to_product_mappings;
-- avg_sim should be >0.65, pct_above_0_7 should be >40%
```

---

## 🚨 Troubleshooting

### Issue: Rate limit errors

**Error**: `429 Too Many Requests`

**Solution**:
- Add `time.sleep(0.05)` between queries (50ms delay)
- Use batching approach (Option 2)
- Reduce `max_workers` in parallel version

---

### Issue: Low similarity scores

**Problem**: Average similarity <0.6

**Solutions**:
1. Check if using correct embeddings (image vs text)
2. Verify DF2 embeddings are normalized
3. Try different Vector Search index (hybrid?)
4. Sample-inspect low-scoring matches

---

### Issue: Category misalignment

**Problem**: DF2 tops mapping to shoes

**Solutions**:
1. Add category filters to Vector Search query
2. Post-filter mappings by category rules
3. Re-run with category constraints

---

## 📋 Next Steps

Once Phase 2 completes:

1. ✅ ~110K mappings created
2. ✅ Quality metrics acceptable (avg sim >0.65)
3. ✅ Manual review: 70%+ positive

**→ Proceed to Phase 3**: `GRAPH_CONSTRUCTION_GUIDE.md`

**What Phase 3 does**:
- Build NetworkX graph using mappings
- Add co-occurrence edges
- Add similarity edges
- Persist to Delta tables

---

**Phase 2 Complete!** ✅ Ready for graph construction.
