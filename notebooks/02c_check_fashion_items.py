# Databricks notebook source
# MAGIC %md
# MAGIC # Check fashion_items_embeddings for Valid Embeddings
# MAGIC
# MAGIC **Goal**: Determine if fashion_items_embeddings has unique, valid embeddings

# COMMAND ----------

import numpy as np
from pyspark.sql.functions import col, size

print("="*70)
print("CHECKING fashion_items_embeddings")
print("="*70)

# COMMAND ----------

# Check schema
print("\n1. Schema:")
print("-"*70)
spark.sql("DESCRIBE main.fashion_demo.fashion_items_embeddings").show(truncate=False)

# COMMAND ----------

# Get statistics
print("\n2. Statistics:")
print("-"*70)

stats = spark.sql("""
    SELECT
        COUNT(*) as total_rows,
        COUNT(DISTINCT item_id) as unique_items,
        COUNT(clip_embedding) as has_embedding,
        COUNT(CASE WHEN SIZE(clip_embedding) = 512 THEN 1 END) as valid_512d,
        COUNT(CASE WHEN SIZE(clip_embedding) > 0 THEN 1 END) as non_empty,
        COUNT(image_path) as has_image_path
    FROM main.fashion_demo.fashion_items_embeddings
""").first()

print(f"Total rows:        {stats.total_rows:>10,}")
print(f"Unique items:      {stats.unique_items:>10,}")
print(f"Has embeddings:    {stats.has_embedding:>10,}")
print(f"Valid 512D:        {stats.valid_512d:>10,}")
print(f"Non-empty:         {stats.non_empty:>10,}")
print(f"Has image paths:   {stats.has_image_path:>10,}")

# COMMAND ----------

# Sample records
print("\n3. Sample Records:")
print("-"*70)

spark.sql("""
    SELECT item_id, image_path, SIZE(clip_embedding) as emb_size
    FROM main.fashion_demo.fashion_items_embeddings
    LIMIT 10
""").show(truncate=False)

# COMMAND ----------

# CHECK EMBEDDING DIVERSITY
print("\n4. CRITICAL: Embedding Diversity Check")
print("-"*70)

# Get 100 embeddings
sample = spark.sql("""
    SELECT item_id, clip_embedding
    FROM main.fashion_demo.fashion_items_embeddings
    WHERE clip_embedding IS NOT NULL
      AND SIZE(clip_embedding) = 512
    LIMIT 100
""").collect()

if len(sample) == 0:
    print("❌ No valid embeddings found!")
    dbutils.notebook.exit("ERROR: No valid embeddings")

print(f"Retrieved {len(sample)} embeddings for testing")

# Convert to numpy
emb_arrays = [np.array(row.clip_embedding) for row in sample]

# Check if all identical
first_emb = emb_arrays[0]
all_same = all(np.allclose(emb, first_emb, rtol=1e-9) for emb in emb_arrays)

print(f"\nFirst embedding stats:")
print(f"  Sum: {np.sum(first_emb):.6f}")
print(f"  L2 norm: {np.linalg.norm(first_emb):.6f}")
print(f"  Min value: {np.min(first_emb):.6f}")
print(f"  Max value: {np.max(first_emb):.6f}")
print(f"  First 10: {first_emb[:10]}")

if all_same:
    print("\n❌ PROBLEM: All embeddings are IDENTICAL")
    print("   This table has the same issue!")
else:
    print("\n✅ SUCCESS: Embeddings are UNIQUE and DIVERSE!")

    # Calculate diversity metrics
    similarities = []
    for i in range(1, min(20, len(emb_arrays))):
        sim = np.dot(first_emb, emb_arrays[i])
        similarities.append(sim)

    print(f"\nSimilarity to first embedding (19 samples):")
    print(f"  Mean:   {np.mean(similarities):.4f}")
    print(f"  Median: {np.median(similarities):.4f}")
    print(f"  Std:    {np.std(similarities):.4f}")
    print(f"  Range:  [{np.min(similarities):.4f}, {np.max(similarities):.4f}]")

# COMMAND ----------

# CHECK IF EMBEDDINGS ARE ALL ZEROS
print("\n5. Check for Zero Embeddings:")
print("-"*70)

# Check if embeddings are all zeros
zero_check = spark.sql("""
    SELECT
        COUNT(*) as total,
        COUNT(CASE
            WHEN AGGREGATE(clip_embedding, 0.0, (acc, x) -> acc + ABS(x)) = 0.0
            THEN 1
        END) as all_zeros
    FROM main.fashion_demo.fashion_items_embeddings
    WHERE clip_embedding IS NOT NULL
""").first()

print(f"Total embeddings:     {zero_check.total:>10,}")
print(f"All-zero embeddings:  {zero_check.all_zeros:>10,}")

if zero_check.all_zeros > 0:
    print(f"\n⚠️  WARNING: {zero_check.all_zeros} embeddings are all zeros")
else:
    print(f"\n✅ No zero embeddings found")

# COMMAND ----------

# CHECK IMAGE PATHS
print("\n6. Image Path Analysis:")
print("-"*70)

path_sample = spark.sql("""
    SELECT item_id, image_path
    FROM main.fashion_demo.fashion_items_embeddings
    WHERE image_path IS NOT NULL
    LIMIT 10
""").collect()

print("Sample image paths:")
for row in path_sample[:5]:
    print(f"  {row.item_id}: {row.image_path}")

# Check if paths point to DeepFashion2 or products
df2_pattern_count = spark.sql("""
    SELECT COUNT(*) as count
    FROM main.fashion_demo.fashion_items_embeddings
    WHERE image_path LIKE '%deepfashion2%' OR image_path LIKE '%fashion-dataset%'
""").first().count

product_pattern_count = spark.sql("""
    SELECT COUNT(*) as count
    FROM main.fashion_demo.fashion_items_embeddings
    WHERE image_path LIKE '%images/%' AND image_path NOT LIKE '%deepfashion%'
""").first().count

print(f"\nImage path patterns:")
print(f"  DeepFashion2 paths: {df2_pattern_count:>10,}")
print(f"  Product paths:      {product_pattern_count:>10,}")

# COMMAND ----------

# RECOMMENDATION
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if not all_same and stats.valid_512d > 10000:
    print(f"""
✅ fashion_items_embeddings looks GOOD!

Stats:
  • {stats.valid_512d:,} valid 512D embeddings
  • Embeddings are UNIQUE and DIVERSE
  • {'Has' if stats.has_image_path > 0 else 'Missing'} image paths

This can be our source for df2_working_set!

Next: Use fashion_items_embeddings to rebuild df2_working_set
""")
else:
    print(f"""
❌ fashion_items_embeddings has issues:
  • {'All embeddings identical' if all_same else 'OK'}
  • {f'Only {stats.valid_512d:,} valid embeddings' if stats.valid_512d < 10000 else 'OK'}

Next: Check df2_items table or other sources
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check df2_items too

# COMMAND ----------

print("="*70)
print("CHECKING df2_items")
print("="*70)

# Schema
print("\nSchema:")
spark.sql("DESCRIBE main.fashion_demo.df2_items").show(truncate=False)

# Stats
df2_stats = spark.sql("""
    SELECT
        COUNT(*) as total_rows,
        COUNT(item_uid) as has_item_uid,
        COUNT(clip_embedding) as has_embedding,
        COUNT(CASE WHEN SIZE(clip_embedding) = 512 THEN 1 END) as valid_512d,
        COUNT(image_path) as has_image_path
    FROM main.fashion_demo.df2_items
""").first()

print(f"\ndf2_items Stats:")
print(f"  Total rows:        {df2_stats.total_rows:>10,}")
print(f"  Has item_uid:      {df2_stats.has_item_uid:>10,}")
print(f"  Has embeddings:    {df2_stats.has_embedding:>10,}")
print(f"  Valid 512D:        {df2_stats.valid_512d:>10,}")
print(f"  Has image paths:   {df2_stats.has_image_path:>10,}")

# Sample
print("\nSample:")
spark.sql("""
    SELECT item_uid, image_path, SIZE(clip_embedding) as emb_size
    FROM main.fashion_demo.df2_items
    LIMIT 5
""").show(truncate=False)

if df2_stats.valid_512d > 0:
    print(f"\n✅ df2_items has {df2_stats.valid_512d:,} valid embeddings!")
    print("   Should check if these are diverse too...")
else:
    print(f"\n❌ df2_items has no valid embeddings")
