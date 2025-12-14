# Databricks notebook source
# MAGIC %md
# MAGIC # DIAGNOSTIC: DeepFashion2 Embedding Issues
# MAGIC
# MAGIC **Problem**: All DF2 items returning same top-5 products
# MAGIC **Hypothesis**: Wrong embeddings in df2_working_set OR embeddings are all identical
# MAGIC
# MAGIC Let's investigate!

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Compare Table Schemas

# COMMAND ----------

print("="*70)
print("SCHEMA COMPARISON")
print("="*70)

# Check df2_working_set
print("\n1. df2_working_set (current source):")
print("-"*70)
spark.sql("DESCRIBE main.fashion_demo.df2_working_set").show(truncate=False)

# Check df2_items
print("\n2. df2_items (potential correct source):")
print("-"*70)
try:
    spark.sql("DESCRIBE main.fashion_demo.df2_items").show(truncate=False)
except:
    print("⚠️  Table does not exist")

# Check fashion_items_embeddings
print("\n3. fashion_items_embeddings:")
print("-"*70)
try:
    spark.sql("DESCRIBE main.fashion_demo.fashion_items_embeddings").show(truncate=False)
except:
    print("⚠️  Table does not exist")

# Check deepfashion2_clip_embeddings (original source)
print("\n4. deepfashion2_clip_embeddings (original):")
print("-"*70)
spark.sql("DESCRIBE main.fashion_demo.deepfashion2_clip_embeddings").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Sample and Compare Embeddings

# COMMAND ----------

from pyspark.sql.functions import col, size
import numpy as np

# Sample from df2_working_set
print("Sample from df2_working_set:")
df2_ws_sample = spark.sql("""
    SELECT item_uid, filename, clip_embedding
    FROM main.fashion_demo.df2_working_set
    LIMIT 5
""")
df2_ws_sample.select("item_uid", "filename", size("clip_embedding").alias("emb_size")).show(truncate=False)

# Get first embedding for comparison
first_emb_ws = df2_ws_sample.select("clip_embedding").first()[0]
print(f"\nFirst embedding from df2_working_set:")
print(f"  Length: {len(first_emb_ws)}")
print(f"  First 10 values: {first_emb_ws[:10]}")
print(f"  Sum: {sum(first_emb_ws):.6f}")
print(f"  L2 norm: {np.linalg.norm(first_emb_ws):.6f}")

# COMMAND ----------

# Sample from original deepfashion2_clip_embeddings
print("\nSample from deepfashion2_clip_embeddings (original):")
df2_orig_sample = spark.sql("""
    SELECT item_uid, filename, clip_embedding
    FROM main.fashion_demo.deepfashion2_clip_embeddings
    LIMIT 5
""")
df2_orig_sample.select("item_uid", "filename", size("clip_embedding").alias("emb_size")).show(truncate=False)

# Get first embedding
first_emb_orig = df2_orig_sample.select("clip_embedding").first()[0]
print(f"\nFirst embedding from deepfashion2_clip_embeddings:")
print(f"  Length: {len(first_emb_orig)}")
print(f"  First 10 values: {first_emb_orig[:10]}")
print(f"  Sum: {sum(first_emb_orig):.6f}")
print(f"  L2 norm: {np.linalg.norm(first_emb_orig):.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Check if Embeddings are Identical

# COMMAND ----------

# Check if all embeddings in df2_working_set are the same
print("Checking embedding uniqueness in df2_working_set...")

# Get first 100 embeddings
embeddings_ws = spark.sql("""
    SELECT item_uid, clip_embedding
    FROM main.fashion_demo.df2_working_set
    LIMIT 100
""").collect()

# Convert to numpy arrays
emb_arrays = [np.array(row.clip_embedding) for row in embeddings_ws]

# Check if they're all identical
first_emb = emb_arrays[0]
all_same = all(np.allclose(emb, first_emb) for emb in emb_arrays)

print(f"\n{'⚠️  PROBLEM' if all_same else '✅ OK'}: All 100 embeddings are {'IDENTICAL' if all_same else 'DIFFERENT'}")

if all_same:
    print("\n🚨 This explains why all items return the same top-5 products!")
    print("   All DF2 items have identical embeddings, so Vector Search returns same results.")
else:
    # Check diversity
    similarities = []
    for i in range(1, min(10, len(emb_arrays))):
        sim = np.dot(emb_arrays[0], emb_arrays[i])
        similarities.append(sim)

    print(f"\nSimilarity to first embedding (sample of 10):")
    print(f"  Mean: {np.mean(similarities):.6f}")
    print(f"  Min:  {np.min(similarities):.6f}")
    print(f"  Max:  {np.max(similarities):.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Check Image Paths

# COMMAND ----------

# Check where image paths point to
print("Checking image paths in different tables...")

print("\n1. df2_working_set:")
spark.sql("""
    SELECT item_uid, filename
    FROM main.fashion_demo.df2_working_set
    LIMIT 5
""").show(truncate=False)

print("\n2. deepfashion2_clip_embeddings:")
spark.sql("""
    SELECT item_uid, filename, image_path
    FROM main.fashion_demo.deepfashion2_clip_embeddings
    LIMIT 5
""").show(truncate=False)

# Check if df2_items exists and has different image paths
try:
    print("\n3. df2_items:")
    spark.sql("""
        SELECT item_uid, image_path
        FROM main.fashion_demo.df2_items
        LIMIT 5
    """).show(truncate=False)
except:
    print("⚠️  df2_items table does not exist")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Root Cause Analysis

# COMMAND ----------

print("="*70)
print("ROOT CAUSE ANALYSIS")
print("="*70)

# Check: Are we using product embeddings instead of DF2 embeddings?
print("\n1. Testing if embeddings match product catalog...")

# Get a sample product embedding
product_emb = spark.sql("""
    SELECT product_id, hybrid_embedding
    FROM main.fashion_demo.products_working_set
    LIMIT 1
""").first()[1]

# Compare with DF2 embedding
df2_emb = embeddings_ws[0].clip_embedding

if np.allclose(np.array(product_emb), np.array(df2_emb)):
    print("🚨 DF2 embeddings are PRODUCT embeddings!")
    print("   This is the wrong data source.")
else:
    print("✅ DF2 embeddings are NOT from product catalog")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Solution: Rebuild df2_working_set

# COMMAND ----------

print("="*70)
print("SOLUTION")
print("="*70)

# Check what tables we have available
print("\nAvailable DeepFashion2 tables:")

tables = spark.sql("""
    SHOW TABLES IN main.fashion_demo
""").filter("tableName LIKE '%fashion%' OR tableName LIKE '%df2%'").collect()

for table in tables:
    table_name = table.tableName
    try:
        count = spark.table(f"main.fashion_demo.{table_name}").count()
        print(f"  • {table_name:50s} {count:>10,} rows")
    except:
        print(f"  • {table_name:50s} {'ERROR':>10}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Proposed Fix

# COMMAND ----------

print("="*70)
print("PROPOSED FIX")
print("="*70)

print("""
Based on investigation, we should:

1. **Identify correct DeepFashion2 source**:
   - Check if `df2_items` has correct outfit embeddings
   - Or use `fashion_items_embeddings` if embeddings are valid
   - Fall back to re-generating embeddings if needed

2. **Recreate df2_working_set**:
   CREATE OR REPLACE TABLE main.fashion_demo.df2_working_set AS
   SELECT
       item_uid,
       clip_embedding,  -- FROM CORRECT SOURCE
       image_path,
       filename
   FROM [CORRECT_SOURCE_TABLE]
   WHERE clip_embedding IS NOT NULL
     AND SIZE(clip_embedding) = 512

3. **Verify diversity**:
   - Check that embeddings are unique
   - Verify image paths point to DeepFashion2 images (not products)

4. **Re-run Phase 2 mapping**:
   - With correct DF2 embeddings
   - Should get diverse top-5 results
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Next Steps

# COMMAND ----------

# Let's check df2_items if it exists
try:
    print("Checking df2_items table...")
    df2_items_info = spark.sql("""
        SELECT
            COUNT(*) as row_count,
            COUNT(clip_embedding) as has_embedding,
            COUNT(CASE WHEN SIZE(clip_embedding) = 512 THEN 1 END) as valid_512d,
            COUNT(image_path) as has_image_path
        FROM main.fashion_demo.df2_items
    """).first()

    print(f"\ndf2_items statistics:")
    print(f"  Total rows:        {df2_items_info.row_count:>10,}")
    print(f"  Has embeddings:    {df2_items_info.has_embedding:>10,}")
    print(f"  Valid 512D:        {df2_items_info.valid_512d:>10,}")
    print(f"  Has image paths:   {df2_items_info.has_image_path:>10,}")

    if df2_items_info.valid_512d > 0:
        print("\n✅ df2_items looks promising! Let's sample embeddings...")

        # Sample embeddings from df2_items
        df2_items_sample = spark.sql("""
            SELECT item_uid, clip_embedding, image_path
            FROM main.fashion_demo.df2_items
            WHERE clip_embedding IS NOT NULL
              AND SIZE(clip_embedding) = 512
            LIMIT 10
        """).collect()

        # Check diversity
        item_embs = [np.array(row.clip_embedding) for row in df2_items_sample]
        if len(item_embs) > 1:
            all_same = all(np.allclose(emb, item_embs[0]) for emb in item_embs[1:])
            print(f"\n  Embeddings unique: {'NO - ALL SAME' if all_same else 'YES - DIFFERENT ✅'}")

            if not all_same:
                print("\n🎉 df2_items has valid, unique embeddings!")
                print("   We should use this as our source.")

except Exception as e:
    print(f"❌ Error checking df2_items: {e}")
    print("\nWe may need to use a different table or regenerate embeddings.")
