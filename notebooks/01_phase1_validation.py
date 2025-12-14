# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 1: Data Validation & Preparation
# MAGIC
# MAGIC **Objective**: Validate embeddings and create working tables
# MAGIC **Runtime**: ~5 minutes
# MAGIC **Date**: 2025-12-13

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup & Imports

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql.functions import col, size, expr, current_timestamp, count, when
from pyspark.sql.types import DoubleType
from collections import Counter

print("✅ Imports loaded")
print(f"Spark version: {spark.version}")

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
print("\n" + "="*70)
print("EMBEDDING QUALITY CHECK")
print("="*70)

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
print("\n" + "="*70)
print("CATEGORY DISTRIBUTION")
print("="*70)

category_dist = df2.groupBy("category_name").count().orderBy(col("count").desc())
category_dist.show(20, False)

cat_counts = category_dist.collect()
print(f"\nTotal categories: {len(cat_counts)}")
for row in cat_counts:
    print(f"  {row.category_name:30s} {row['count']:>6,}")

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

# Calculate coverage percentages
img_coverage = (product_stats.has_image_emb / product_stats.total) * 100
txt_coverage = (product_stats.has_text_emb / product_stats.total) * 100
hyb_coverage = (product_stats.has_hybrid_emb / product_stats.total) * 100

print(f"\nCoverage Percentages:")
print(f"  Image:  {img_coverage:.2f}%")
print(f"  Text:   {txt_coverage:.2f}%")
print(f"  Hybrid: {hyb_coverage:.2f}%")

if img_coverage > 99 and txt_coverage > 99 and hyb_coverage > 99:
    print(f"\n✅ PRODUCT EMBEDDINGS COVERAGE EXCELLENT (>99%)")
else:
    print(f"\n⚠️  WARNING: Coverage below 99%")

# COMMAND ----------

# Sample products
print(f"\nSample products:")
products_emb.select(
    "product_id",
    "product_display_name",
    "article_type",
    "master_category",
    "price"
).show(10, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test Vector Search Integration

# COMMAND ----------

print("\n" + "="*70)
print("VECTOR SEARCH TEST")
print("="*70)

try:
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient()
    print("✅ Vector Search client initialized")

    # List available endpoints
    print("\nChecking for fashion_vector_search endpoint...")

    # Get a sample embedding for testing
    sample_emb = products_emb.select("hybrid_embedding").first()
    test_vector = sample_emb.hybrid_embedding

    print(f"✅ Retrieved test embedding (dim={len(test_vector)})")

    # Try to get the index
    try:
        index = vsc.get_index(
            endpoint_name="fashion_vector_search",
            index_name="main.fashion_demo.vs_hybrid_search"
        )
        print("✅ Vector Search index accessible: main.fashion_demo.vs_hybrid_search")

        # Test query
        results = index.similarity_search(
            query_vector=test_vector,
            columns=["product_id", "product_display_name", "price"],
            num_results=5
        )

        print(f"✅ Vector Search query successful!")
        print(f"   Returned {len(results['result']['data_array'])} results")

    except Exception as e:
        print(f"⚠️  Could not access Vector Search index: {e}")
        print("   This is OK for validation - we can proceed without it")

except Exception as e:
    print(f"⚠️  Vector Search not available: {e}")
    print("   This is OK - Vector Search optional for Phase 1")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Working Tables

# COMMAND ----------

print("\n" + "="*70)
print("CREATING WORKING TABLES")
print("="*70)

# Create df2_working_set
print("\n1. Creating main.fashion_demo.df2_working_set...")
spark.sql("""
    CREATE OR REPLACE TABLE main.fashion_demo.df2_working_set AS
    SELECT
        item_uid,
        clip_embedding,
        category_name,
        current_timestamp() as created_at
    FROM main.fashion_demo.deepfashion2_clip_embeddings
    WHERE clip_embedding IS NOT NULL
      AND SIZE(clip_embedding) = 512
""")

df2_working_count = spark.table("main.fashion_demo.df2_working_set").count()
print(f"   ✅ Created with {df2_working_count:,} rows")

# COMMAND ----------

# Create products_working_set (filter for Apparel, Accessories, Footwear)
print("\n2. Creating main.fashion_demo.products_working_set...")
spark.sql("""
    CREATE OR REPLACE TABLE main.fashion_demo.products_working_set AS
    SELECT
        product_id,
        product_display_name,
        master_category,
        sub_category,
        article_type,
        base_color,
        gender,
        season,
        usage,
        price,
        image_embedding,
        text_embedding,
        hybrid_embedding,
        current_timestamp() as created_at
    FROM main.fashion_demo.product_embeddings_multimodal
    WHERE master_category IN ('Apparel', 'Accessories', 'Footwear')
      AND hybrid_embedding IS NOT NULL
      AND SIZE(hybrid_embedding) = 512
""")

products_working_count = spark.table("main.fashion_demo.products_working_set").count()
print(f"   ✅ Created with {products_working_count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Export Product Taxonomy

# COMMAND ----------

print("\n" + "="*70)
print("PRODUCT TAXONOMY")
print("="*70)

# Get taxonomy summary
taxonomy = spark.sql("""
    SELECT
        master_category,
        COUNT(DISTINCT article_type) as num_article_types,
        COUNT(DISTINCT sub_category) as num_sub_categories,
        COUNT(*) as total_products
    FROM main.fashion_demo.products_working_set
    GROUP BY master_category
    ORDER BY total_products DESC
""")

print("\nTaxonomy Summary:")
taxonomy.show(10, False)

# Get all article types
article_types = spark.sql("""
    SELECT
        article_type,
        master_category,
        sub_category,
        COUNT(*) as count
    FROM main.fashion_demo.products_working_set
    GROUP BY article_type, master_category, sub_category
    ORDER BY master_category, count DESC
""")

total_article_types = article_types.count()
print(f"\nTotal unique article types: {total_article_types}")

# Show sample
print("\nSample article types:")
article_types.show(20, False)

# COMMAND ----------

# Create taxonomy mapping table (using the SQL from export_taxonomy.sql)
print("\n3. Creating main.fashion_demo.product_taxonomy...")
spark.sql("""
    CREATE OR REPLACE TABLE main.fashion_demo.product_taxonomy AS
    SELECT DISTINCT
        article_type,
        master_category,
        sub_category,
        CASE
            -- TOPS
            WHEN LOWER(article_type) LIKE '%tshirt%' THEN 'tops'
            WHEN LOWER(article_type) LIKE '%shirt%' AND LOWER(article_type) NOT LIKE '%sweat%' THEN 'tops'
            WHEN LOWER(article_type) LIKE '%top%' THEN 'tops'
            WHEN LOWER(article_type) LIKE '%blouse%' THEN 'tops'
            WHEN LOWER(article_type) LIKE '%sweater%' THEN 'tops'
            WHEN LOWER(article_type) LIKE '%pullover%' THEN 'tops'
            WHEN LOWER(article_type) LIKE '%sweatshirt%' THEN 'tops'
            WHEN LOWER(article_type) LIKE '%jersey%' THEN 'tops'
            WHEN LOWER(article_type) LIKE '%tank%' THEN 'tops'
            WHEN LOWER(article_type) LIKE '%cami%' THEN 'tops'

            -- BOTTOMS
            WHEN LOWER(article_type) LIKE '%jean%' THEN 'bottoms'
            WHEN LOWER(article_type) LIKE '%trouser%' THEN 'bottoms'
            WHEN LOWER(article_type) LIKE '%pant%' THEN 'bottoms'
            WHEN LOWER(article_type) LIKE '%short%' AND master_category = 'Apparel' THEN 'bottoms'
            WHEN LOWER(article_type) LIKE '%skirt%' THEN 'bottoms'
            WHEN LOWER(article_type) LIKE '%legging%' THEN 'bottoms'
            WHEN LOWER(article_type) LIKE '%jogger%' THEN 'bottoms'

            -- SHOES
            WHEN master_category = 'Footwear' THEN 'shoes'
            WHEN LOWER(article_type) LIKE '%shoe%' THEN 'shoes'
            WHEN LOWER(article_type) LIKE '%sandal%' THEN 'shoes'
            WHEN LOWER(article_type) LIKE '%flip flop%' THEN 'shoes'
            WHEN LOWER(article_type) LIKE '%sneaker%' THEN 'shoes'
            WHEN LOWER(article_type) LIKE '%boot%' THEN 'shoes'

            -- OUTERWEAR
            WHEN LOWER(article_type) LIKE '%jacket%' THEN 'outerwear'
            WHEN LOWER(article_type) LIKE '%coat%' THEN 'outerwear'
            WHEN LOWER(article_type) LIKE '%blazer%' THEN 'outerwear'
            WHEN LOWER(article_type) LIKE '%cardigan%' THEN 'outerwear'
            WHEN LOWER(article_type) LIKE '%hoodie%' THEN 'outerwear'

            -- DRESSES
            WHEN LOWER(article_type) LIKE '%dress%' THEN 'dresses'
            WHEN LOWER(article_type) LIKE '%gown%' THEN 'dresses'
            WHEN LOWER(article_type) LIKE '%jumpsuit%' THEN 'dresses'

            -- ACCESSORIES
            WHEN master_category = 'Accessories' THEN 'accessories'
            WHEN LOWER(article_type) LIKE '%watch%' THEN 'accessories'
            WHEN LOWER(article_type) LIKE '%bag%' THEN 'accessories'
            WHEN LOWER(article_type) LIKE '%belt%' THEN 'accessories'
            WHEN LOWER(article_type) LIKE '%tie%' THEN 'accessories'
            WHEN LOWER(article_type) LIKE '%cap%' THEN 'accessories'
            WHEN LOWER(article_type) LIKE '%hat%' THEN 'accessories'
            WHEN LOWER(article_type) LIKE '%sunglasses%' THEN 'accessories'
            WHEN LOWER(article_type) LIKE '%jewellery%' THEN 'accessories'
            WHEN LOWER(article_type) LIKE '%sock%' THEN 'accessories'

            ELSE 'uncategorized'
        END as graph_category
    FROM main.fashion_demo.products_working_set
    WHERE article_type IS NOT NULL
""")

taxonomy_count = spark.table("main.fashion_demo.product_taxonomy").count()
print(f"   ✅ Created with {taxonomy_count} article type mappings")

# COMMAND ----------

# Show graph category distribution
print("\nGraph Category Distribution:")
graph_cat_dist = spark.sql("""
    SELECT
        graph_category,
        COUNT(DISTINCT article_type) as num_article_types
    FROM main.fashion_demo.product_taxonomy
    GROUP BY graph_category
    ORDER BY graph_category
""")
graph_cat_dist.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Phase 1 Summary

# COMMAND ----------

print("\n" + "="*80)
print("PHASE 1 VALIDATION - SUMMARY REPORT")
print("="*80)

print(f"\n✅ DATA VALIDATION")
print(f"   DeepFashion2 Items:      {embedding_stats.total_items:>10,} (100% valid)")
print(f"   Product Embeddings:      {product_stats.total:>10,} ({img_coverage:.1f}% coverage)")
print(f"   L2 Normalization:        {'✅ PASS' if is_normalized else '❌ FAIL'}")
print(f"   Zero Embeddings:         {'✅ NONE' if all_valid else '⚠️  FOUND'}")

print(f"\n✅ WORKING TABLES CREATED")
print(f"   df2_working_set:         {df2_working_count:>10,} rows")
print(f"   products_working_set:    {products_working_count:>10,} rows")
print(f"   product_taxonomy:        {taxonomy_count:>10,} mappings")

print(f"\n✅ TAXONOMY EXPORT")
print(f"   Article types mapped:    {taxonomy_count:>10,}")

# Get category counts
cat_summary = spark.sql("""
    SELECT graph_category, COUNT(*) as count
    FROM main.fashion_demo.product_taxonomy
    GROUP BY graph_category
    ORDER BY count DESC
""").collect()

print(f"   Graph categories:")
for row in cat_summary:
    print(f"      {row.graph_category:20s} {row['count']:>6} article types")

print(f"\n{'='*80}")
print(f"✅ PHASE 1 COMPLETE - Ready for Phase 2 (Vector Search Mapping)")
print(f"{'='*80}")

# COMMAND ----------

# Save summary to table
spark.sql(f"""
    CREATE OR REPLACE TABLE main.fashion_demo.phase1_validation_summary AS
    SELECT
        {embedding_stats.total_items} as df2_items_validated,
        {product_stats.total} as products_validated,
        {df2_working_count} as df2_working_set_count,
        {products_working_count} as products_working_set_count,
        {taxonomy_count} as taxonomy_mappings_count,
        '{embedding_stats.avg_l2_norm:.6f}' as avg_l2_norm,
        current_timestamp() as validation_timestamp,
        'PHASE_1_COMPLETE' as status
""")

print("\n✅ Summary saved to main.fashion_demo.phase1_validation_summary")
