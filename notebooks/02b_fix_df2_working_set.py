# Databricks notebook source
# MAGIC %md
# MAGIC # FIX: Rebuild df2_working_set with Correct Embeddings
# MAGIC
# MAGIC **Goal**: Create df2_working_set from the correct DeepFashion2 source
# MAGIC **Based on**: Findings from diagnostic notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# CONFIGURATION - UPDATE BASED ON DIAGNOSTIC FINDINGS
SOURCE_TABLE = "main.fashion_demo.df2_items"  # Change if needed
OUTPUT_TABLE = "main.fashion_demo.df2_working_set"

print(f"Source Table: {SOURCE_TABLE}")
print(f"Output Table: {OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Validate Source Table

# COMMAND ----------

from pyspark.sql.functions import col, size, count
import numpy as np

print("="*70)
print("SOURCE TABLE VALIDATION")
print("="*70)

# Check source exists
try:
    source_df = spark.table(SOURCE_TABLE)
    print(f"✅ Source table exists: {SOURCE_TABLE}")
except:
    print(f"❌ Source table not found: {SOURCE_TABLE}")
    dbutils.notebook.exit("ERROR: Source table not found")

# Check schema
print("\nSchema:")
source_df.printSchema()

# Get statistics
stats = source_df.selectExpr(
    "COUNT(*) as total_rows",
    "COUNT(clip_embedding) as has_embedding",
    "COUNT(CASE WHEN SIZE(clip_embedding) = 512 THEN 1 END) as valid_512d",
    "COUNT(image_path) as has_image_path"
).first()

print(f"\nStatistics:")
print(f"  Total rows:        {stats.total_rows:>10,}")
print(f"  Has embeddings:    {stats.has_embedding:>10,}")
print(f"  Valid 512D:        {stats.valid_512d:>10,}")
print(f"  Has image paths:   {stats.has_image_path:>10,}")

# Validate embeddings are unique
print("\nValidating embedding uniqueness...")
sample_embeddings = source_df.select("clip_embedding").filter(
    "clip_embedding IS NOT NULL AND SIZE(clip_embedding) = 512"
).limit(20).collect()

if len(sample_embeddings) < 2:
    print("❌ Not enough embeddings to validate")
    dbutils.notebook.exit("ERROR: Insufficient embeddings")

emb_arrays = [np.array(row.clip_embedding) for row in sample_embeddings]
first_emb = emb_arrays[0]
all_same = all(np.allclose(emb, first_emb) for emb in emb_arrays[1:])

if all_same:
    print("❌ All embeddings are IDENTICAL - this won't work!")
    dbutils.notebook.exit("ERROR: Embeddings are not unique")
else:
    print("✅ Embeddings are UNIQUE and diverse")

    # Calculate some similarity stats
    similarities = [np.dot(first_emb, emb) for emb in emb_arrays[1:]]
    print(f"   Mean similarity to first: {np.mean(similarities):.4f}")
    print(f"   Range: [{np.min(similarities):.4f}, {np.max(similarities):.4f}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create New df2_working_set

# COMMAND ----------

print("="*70)
print("CREATING NEW df2_working_set")
print("="*70)

# Build the new working set
create_sql = f"""
CREATE OR REPLACE TABLE {OUTPUT_TABLE} AS
SELECT
    item_uid,
    clip_embedding,
    image_path,
    filename,
    current_timestamp() as created_at
FROM {SOURCE_TABLE}
WHERE clip_embedding IS NOT NULL
  AND SIZE(clip_embedding) = 512
"""

print("\nExecuting SQL:")
print(create_sql)
print()

spark.sql(create_sql)

# Verify
new_count = spark.table(OUTPUT_TABLE).count()
print(f"✅ Created {OUTPUT_TABLE} with {new_count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validate New Working Set

# COMMAND ----------

print("="*70)
print("VALIDATION OF NEW WORKING SET")
print("="*70)

# Check schema
print("\nNew schema:")
spark.table(OUTPUT_TABLE).printSchema()

# Get statistics
new_stats = spark.sql(f"""
    SELECT
        COUNT(*) as total_rows,
        COUNT(DISTINCT item_uid) as unique_items,
        COUNT(clip_embedding) as has_embedding,
        COUNT(CASE WHEN SIZE(clip_embedding) = 512 THEN 1 END) as valid_512d,
        COUNT(image_path) as has_image_path,
        COUNT(filename) as has_filename
    FROM {OUTPUT_TABLE}
""").first()

print(f"\nStatistics:")
print(f"  Total rows:        {new_stats.total_rows:>10,}")
print(f"  Unique items:      {new_stats.unique_items:>10,}")
print(f"  Has embeddings:    {new_stats.has_embedding:>10,}")
print(f"  Valid 512D:        {new_stats.valid_512d:>10,}")
print(f"  Has image paths:   {new_stats.has_image_path:>10,}")
print(f"  Has filenames:     {new_stats.has_filename:>10,}")

# Sample data
print("\nSample records:")
spark.table(OUTPUT_TABLE).select(
    "item_uid", "filename", "image_path", size("clip_embedding").alias("emb_size")
).show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Final Embedding Diversity Check

# COMMAND ----------

print("="*70)
print("FINAL DIVERSITY CHECK")
print("="*70)

# Get 100 random embeddings
test_embeddings = spark.table(OUTPUT_TABLE).select("item_uid", "clip_embedding").limit(100).collect()

emb_test_arrays = [np.array(row.clip_embedding) for row in test_embeddings]

# Check if all same
first_test = emb_test_arrays[0]
all_same_test = all(np.allclose(emb, first_test) for emb in emb_test_arrays[1:])

if all_same_test:
    print("❌ PROBLEM: Embeddings are still identical!")
    print("   Phase 2 will not work correctly.")
else:
    print("✅ EXCELLENT: Embeddings are diverse!")

    # Calculate pairwise similarities for first 10
    similarities_matrix = []
    for i in range(min(10, len(emb_test_arrays))):
        for j in range(i+1, min(10, len(emb_test_arrays))):
            sim = np.dot(emb_test_arrays[i], emb_test_arrays[j])
            similarities_matrix.append(sim)

    print(f"\nPairwise similarities (first 10 items):")
    print(f"  Mean:   {np.mean(similarities_matrix):.4f}")
    print(f"  Median: {np.median(similarities_matrix):.4f}")
    print(f"  Std:    {np.std(similarities_matrix):.4f}")
    print(f"  Range:  [{np.min(similarities_matrix):.4f}, {np.max(similarities_matrix):.4f}]")

    print("\n🎉 df2_working_set is ready for Phase 2 mapping!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test Vector Search with New Embeddings

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

print("="*70)
print("TESTING VECTOR SEARCH")
print("="*70)

# Initialize Vector Search
vsc = VectorSearchClient(disable_notice=True)
vs_index = vsc.get_index(
    endpoint_name="one-env-shared-endpoint-11",
    index_name="main.fashion_demo.vs_product_hybrid_search"
)

print("✅ Connected to Vector Search")

# Test with 3 different DF2 items
print("\nTesting 3 different DF2 items to verify we get DIFFERENT results...\n")

test_items = spark.table(OUTPUT_TABLE).limit(3).collect()

all_results = []
for idx, item in enumerate(test_items, 1):
    print(f"{idx}. Testing item_uid: {item.item_uid}")

    results = vs_index.similarity_search(
        query_vector=item.clip_embedding,
        columns=["product_id", "product_display_name"],
        num_results=5
    )

    if results and 'result' in results and 'data_array' in results['result']:
        top_5_ids = [row[0] for row in results['result']['data_array']]
        print(f"   Top 5 product IDs: {top_5_ids}")
        all_results.append(set(top_5_ids))
    else:
        print(f"   No results returned")

# Check if results are different
if len(all_results) == 3:
    if all_results[0] == all_results[1] == all_results[2]:
        print("\n❌ PROBLEM: All 3 items returned SAME products!")
        print("   This indicates an issue with embeddings or Vector Search.")
    else:
        print("\n✅ SUCCESS: Different items returned DIFFERENT products!")
        print("   Phase 2 mapping will work correctly now.")

        # Show overlap
        overlap_1_2 = len(all_results[0] & all_results[1])
        overlap_1_3 = len(all_results[0] & all_results[2])
        overlap_2_3 = len(all_results[1] & all_results[2])

        print(f"\n   Overlap analysis:")
        print(f"     Items 1 & 2: {overlap_1_2}/5 products in common")
        print(f"     Items 1 & 3: {overlap_1_3}/5 products in common")
        print(f"     Items 2 & 3: {overlap_2_3}/5 products in common")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary

# COMMAND ----------

print("="*70)
print("SUMMARY")
print("="*70)

print(f"""
✅ df2_working_set rebuilt from: {SOURCE_TABLE}

Statistics:
  • {new_stats.total_rows:,} DeepFashion2 items
  • {new_stats.valid_512d:,} valid 512D embeddings
  • Embeddings are UNIQUE and DIVERSE

Next Steps:
  1. Re-run Phase 2 mapping notebook (02_phase2_mapping_test.py)
  2. Verify different DF2 items get different top-5 products
  3. Check image previews to validate quality
  4. If good, run full Phase 2 for all 22K items

Table Ready: {OUTPUT_TABLE}
""")
