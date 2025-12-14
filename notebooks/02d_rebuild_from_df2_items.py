# Databricks notebook source
# MAGIC %md
# MAGIC # Rebuild df2_working_set from df2_items
# MAGIC
# MAGIC **Source**: main.fashion_demo.df2_items (10,617 rows)
# MAGIC **Issues to fix**:
# MAGIC - item_uid is NULL → extract from image_path
# MAGIC - Verify embeddings are diverse
# MAGIC - Populate any missing columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Inspect df2_items

# COMMAND ----------

from pyspark.sql.functions import col, size, regexp_extract, split, element_at
import numpy as np

print("="*70)
print("INSPECTING df2_items")
print("="*70)

# Schema
print("\nSchema:")
spark.sql("DESCRIBE main.fashion_demo.df2_items").show(truncate=False)

# COMMAND ----------

# Statistics
print("\nStatistics:")
stats = spark.sql("""
    SELECT
        COUNT(*) as total_rows,
        COUNT(item_uid) as has_item_uid,
        COUNT(CASE WHEN item_uid IS NOT NULL AND item_uid != '' THEN 1 END) as non_null_uid,
        COUNT(clip_embedding) as has_embedding,
        COUNT(CASE WHEN SIZE(clip_embedding) = 512 THEN 1 END) as valid_512d,
        COUNT(CASE WHEN SIZE(clip_embedding) > 0 THEN 1 END) as non_empty_emb,
        COUNT(image_path) as has_image_path
    FROM main.fashion_demo.df2_items
""").first()

print(f"Total rows:           {stats.total_rows:>10,}")
print(f"Has item_uid:         {stats.has_item_uid:>10,}")
print(f"Non-null item_uid:    {stats.non_null_uid:>10,}")
print(f"Has embeddings:       {stats.has_embedding:>10,}")
print(f"Valid 512D:           {stats.valid_512d:>10,}")
print(f"Non-empty embeddings: {stats.non_empty_emb:>10,}")
print(f"Has image paths:      {stats.has_image_path:>10,}")

# COMMAND ----------

# Sample data
print("\nSample records:")
spark.sql("""
    SELECT item_uid, image_path, SIZE(clip_embedding) as emb_size
    FROM main.fashion_demo.df2_items
    LIMIT 10
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Check Embedding Diversity (CRITICAL)

# COMMAND ----------

print("="*70)
print("EMBEDDING DIVERSITY CHECK")
print("="*70)

# Get 100 sample embeddings
sample = spark.sql("""
    SELECT image_path, clip_embedding
    FROM main.fashion_demo.df2_items
    WHERE clip_embedding IS NOT NULL
      AND SIZE(clip_embedding) = 512
    LIMIT 100
""").collect()

print(f"Retrieved {len(sample)} valid embeddings for testing")

if len(sample) == 0:
    print("\n❌ ERROR: No valid embeddings found in df2_items!")
    dbutils.notebook.exit("ERROR: No valid embeddings")

# Convert to numpy
emb_arrays = [np.array(row.clip_embedding) for row in sample]

# Check first embedding
first_emb = emb_arrays[0]
print(f"\nFirst embedding analysis:")
print(f"  Image: {sample[0].image_path}")
print(f"  Length: {len(first_emb)}")
print(f"  Sum: {np.sum(first_emb):.6f}")
print(f"  L2 norm: {np.linalg.norm(first_emb):.6f}")
print(f"  Min: {np.min(first_emb):.6f}, Max: {np.max(first_emb):.6f}")
print(f"  First 10 values: {first_emb[:10]}")

# Check if all identical
all_same = all(np.allclose(emb, first_emb, rtol=1e-9) for emb in emb_arrays)

if all_same:
    print("\n❌ CRITICAL PROBLEM: All embeddings are IDENTICAL!")
    print("   df2_items has the same issue as deepfashion2_clip_embeddings")
    print("   We'll need to regenerate embeddings using CLIP model")
else:
    print("\n✅ EXCELLENT: Embeddings are UNIQUE and DIVERSE!")

    # Calculate diversity metrics
    similarities = []
    for i in range(1, min(20, len(emb_arrays))):
        sim = np.dot(first_emb, emb_arrays[i])
        similarities.append(sim)

    print(f"\nDiversity metrics (19 pairwise comparisons):")
    print(f"  Mean similarity:   {np.mean(similarities):.4f}")
    print(f"  Median similarity: {np.median(similarities):.4f}")
    print(f"  Std deviation:     {np.std(similarities):.4f}")
    print(f"  Range:             [{np.min(similarities):.4f}, {np.max(similarities):.4f}]")

    if np.std(similarities) > 0.05:
        print(f"\n🎉 Great diversity! These embeddings are good to use.")
    else:
        print(f"\n⚠️  Low diversity - embeddings may be too similar")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Extract item_uid from image_path

# COMMAND ----------

print("="*70)
print("EXTRACTING item_uid FROM image_path")
print("="*70)

# Show sample paths to understand pattern
print("\nSample image paths:")
sample_paths = spark.sql("""
    SELECT image_path
    FROM main.fashion_demo.df2_items
    LIMIT 5
""").collect()

for row in sample_paths:
    print(f"  {row.image_path}")

# Extract filename (without extension) as item_uid
# Pattern: /path/to/12345.jpg → item_uid = "12345"
df2_with_uid = spark.sql("""
    SELECT
        regexp_extract(image_path, '([0-9]+)\\.jpg$', 1) as item_uid,
        image_path,
        clip_embedding
    FROM main.fashion_demo.df2_items
    WHERE clip_embedding IS NOT NULL
      AND SIZE(clip_embedding) = 512
""")

# Check results
print("\nExtracted item_uid samples:")
df2_with_uid.select("item_uid", "image_path").show(10, truncate=False)

# Verify uniqueness
uid_stats = df2_with_uid.selectExpr(
    "COUNT(*) as total",
    "COUNT(DISTINCT item_uid) as unique_uids",
    "COUNT(CASE WHEN item_uid IS NULL OR item_uid = '' THEN 1 END) as null_uids"
).first()

print(f"\nitem_uid extraction results:")
print(f"  Total records:    {uid_stats.total:>10,}")
print(f"  Unique UIDs:      {uid_stats.unique_uids:>10,}")
print(f"  NULL/empty UIDs:  {uid_stats.null_uids:>10,}")

if uid_stats.null_uids > 0:
    print(f"\n⚠️  WARNING: {uid_stats.null_uids} records couldn't extract item_uid")
    print("   These will be filtered out")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create filename column

# COMMAND ----------

# Add filename column (just the filename with extension)
df2_complete = spark.sql("""
    SELECT
        regexp_extract(image_path, '([0-9]+\\.jpg)$', 1) as filename,
        regexp_extract(image_path, '([0-9]+)\\.jpg$', 1) as item_uid,
        image_path,
        clip_embedding
    FROM main.fashion_demo.df2_items
    WHERE clip_embedding IS NOT NULL
      AND SIZE(clip_embedding) = 512
      AND regexp_extract(image_path, '([0-9]+)\\.jpg$', 1) != ''
""")

print("Sample with filename and item_uid:")
df2_complete.select("item_uid", "filename", "image_path").show(5, truncate=False)

final_count = df2_complete.count()
print(f"\nFinal record count: {final_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create New df2_working_set

# COMMAND ----------

print("="*70)
print("CREATING NEW df2_working_set")
print("="*70)

# Create the new working set
df2_complete.createOrReplaceTempView("df2_staging")

spark.sql("""
    CREATE OR REPLACE TABLE main.fashion_demo.df2_working_set AS
    SELECT
        item_uid,
        filename,
        image_path,
        clip_embedding,
        current_timestamp() as created_at
    FROM df2_staging
""")

new_count = spark.table("main.fashion_demo.df2_working_set").count()
print(f"\n✅ Created main.fashion_demo.df2_working_set")
print(f"   Rows: {new_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Validate New Working Set

# COMMAND ----------

print("="*70)
print("VALIDATION OF NEW df2_working_set")
print("="*70)

# Schema
print("\nSchema:")
spark.table("main.fashion_demo.df2_working_set").printSchema()

# Sample
print("\nSample records:")
spark.sql("""
    SELECT item_uid, filename, SIZE(clip_embedding) as emb_size
    FROM main.fashion_demo.df2_working_set
    LIMIT 10
""").show(truncate=False)

# COMMAND ----------

# Final diversity check on new working set
print("\n" + "="*70)
print("FINAL DIVERSITY CHECK ON NEW WORKING SET")
print("="*70)

final_sample = spark.sql("""
    SELECT item_uid, clip_embedding
    FROM main.fashion_demo.df2_working_set
    LIMIT 100
""").collect()

final_embs = [np.array(row.clip_embedding) for row in final_sample]
first_final = final_embs[0]
all_same_final = all(np.allclose(emb, first_final) for emb in final_embs)

if all_same_final:
    print("❌ PROBLEM: Embeddings are still identical!")
    print("   Something went wrong.")
else:
    print("✅ SUCCESS: Embeddings are diverse!")

    final_sims = [np.dot(first_final, emb) for emb in final_embs[1:20]]
    print(f"\nFinal diversity metrics:")
    print(f"  Mean similarity:   {np.mean(final_sims):.4f}")
    print(f"  Std deviation:     {np.std(final_sims):.4f}")
    print(f"  Range:             [{np.min(final_sims):.4f}, {np.max(final_sims):.4f}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test with Vector Search

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

print("="*70)
print("TESTING WITH VECTOR SEARCH")
print("="*70)

# Initialize
vsc = VectorSearchClient(disable_notice=True)
vs_index = vsc.get_index(
    endpoint_name="one-env-shared-endpoint-11",
    index_name="main.fashion_demo.vs_product_hybrid_search"
)
print("✅ Connected to Vector Search\n")

# Test with 3 different items
test_items = spark.table("main.fashion_demo.df2_working_set").limit(3).collect()

print("Testing 3 different DF2 items...\n")
all_results = []

for idx, item in enumerate(test_items, 1):
    print(f"{idx}. Testing item_uid: {item.item_uid}")

    results = vs_index.similarity_search(
        query_vector=item.clip_embedding,
        columns=["product_id", "product_display_name"],
        num_results=5
    )

    if results and 'result' in results and 'data_array' in results['result']:
        top_5_ids = [int(row[0]) for row in results['result']['data_array']]
        top_5_names = [row[1] for row in results['result']['data_array']]

        print(f"   Top 5 product IDs: {top_5_ids}")
        print(f"   Top 5 names: {[n[:30] for n in top_5_names]}")
        all_results.append(set(top_5_ids))
    else:
        print(f"   No results")

print()

# Check if results are different
if len(all_results) == 3:
    if all_results[0] == all_results[1] == all_results[2]:
        print("❌ PROBLEM: All 3 items returned SAME products!")
        print("   Embeddings may still be identical or Vector Search issue")
    else:
        print("✅ SUCCESS: Different items returned DIFFERENT products!")

        # Show overlap
        overlap_1_2 = len(all_results[0] & all_results[1])
        overlap_1_3 = len(all_results[0] & all_results[2])
        overlap_2_3 = len(all_results[1] & all_results[2])

        print(f"\n   Overlap analysis:")
        print(f"     Items 1 & 2: {overlap_1_2}/5 common")
        print(f"     Items 1 & 3: {overlap_1_3}/5 common")
        print(f"     Items 2 & 3: {overlap_2_3}/5 common")

        print("\n🎉 Phase 2 mapping will work correctly now!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary

# COMMAND ----------

print("="*70)
print("SUMMARY")
print("="*70)

print(f"""
✅ df2_working_set rebuilt from df2_items

Source: main.fashion_demo.df2_items
Output: main.fashion_demo.df2_working_set

Changes:
  • Extracted item_uid from image paths
  • Added filename column
  • Filtered to valid 512D embeddings
  • Final count: {new_count:,} items

Validation:
  • Embeddings are {'UNIQUE ✅' if not all_same_final else 'IDENTICAL ❌'}
  • Vector Search returns {'DIFFERENT ✅' if len(all_results) == 3 and not (all_results[0] == all_results[1] == all_results[2]) else 'SAME ❌'} results

Next Steps:
  1. Re-run Phase 2 test notebook (02_phase2_mapping_test.py)
  2. Verify different top-5 results for each DF2 item
  3. Check image previews for quality
  4. Run full Phase 2 for all {new_count:,} items
""")
