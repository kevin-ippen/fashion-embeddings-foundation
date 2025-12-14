# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2: Vector Search Mapping (TEST MODE)
# MAGIC
# MAGIC **Objective**: Map DeepFashion2 items to product catalog using Vector Search
# MAGIC **Test Mode**: Process 100 items first to validate approach
# MAGIC **Runtime**: ~2-3 minutes for test

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# TEST MODE CONFIGURATION
TEST_MODE = True  # Set to False for full run
TEST_LIMIT = 100  # Number of items to test with

# Mapping configuration
TOP_K = 5  # Top-5 similar products per DF2 item
BATCH_SIZE = 20  # Process 20 items at a time

# Vector Search configuration
VS_ENDPOINT = "one-env-shared-endpoint-11"
VS_INDEX = "main.fashion_demo.vs_product_hybrid_search"

print(f"{'='*70}")
print(f"PHASE 2 CONFIGURATION")
print(f"{'='*70}")
print(f"Test Mode:        {'ENABLED' if TEST_MODE else 'DISABLED'}")
print(f"Items to process: {TEST_LIMIT if TEST_MODE else 'ALL (22,000)'}")
print(f"Top-K per item:   {TOP_K}")
print(f"Batch size:       {BATCH_SIZE}")
print(f"VS Endpoint:      {VS_ENDPOINT}")
print(f"VS Index:         {VS_INDEX}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load DeepFashion2 Items

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
import pandas as pd
import numpy as np
from pyspark.sql.functions import col
import time

# Load DF2 items
if TEST_MODE:
    df2_items = spark.sql(f"""
        SELECT item_uid, clip_embedding, filename
        FROM main.fashion_demo.df2_working_set
        LIMIT {TEST_LIMIT}
    """)
else:
    df2_items = spark.table("main.fashion_demo.df2_working_set")

df2_count = df2_items.count()
print(f"✅ Loaded {df2_count:,} DeepFashion2 items")

# Show sample
print("\nSample items:")
df2_items.select("item_uid", "filename").show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize Vector Search

# COMMAND ----------

# Initialize Vector Search client
vsc = VectorSearchClient(disable_notice=True)

# Get the index
vs_index = vsc.get_index(
    endpoint_name=VS_ENDPOINT,
    index_name=VS_INDEX
)

print(f"✅ Connected to Vector Search index: {VS_INDEX}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Batch Mapping

# COMMAND ----------

# Collect items to Python (works for test mode)
items_pd = df2_items.toPandas()

all_mappings = []
total_batches = (len(items_pd) + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Starting batch mapping:")
print(f"  Items: {len(items_pd):,}")
print(f"  Batches: {total_batches}")
print()

# Process in batches
for batch_idx in range(total_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(items_pd))

    print(f"⏳ Batch {batch_idx + 1}/{total_batches}: Items {start_idx + 1}-{end_idx}...", end=" ")

    # Process batch
    for idx in range(start_idx, end_idx):
        row = items_pd.iloc[idx]
        item_uid = row['item_uid']
        embedding = row['clip_embedding']

        try:
            # Query Vector Search
            results = vs_index.similarity_search(
                query_vector=embedding,
                columns=["product_id", "product_display_name", "article_type",
                        "master_category", "price", "image_path"],
                num_results=TOP_K
            )

            # Extract results
            if results and 'result' in results and 'data_array' in results['result']:
                for rank, result_row in enumerate(results['result']['data_array'], 1):
                    all_mappings.append({
                        'df2_item_uid': item_uid,
                        'product_id': int(result_row[0]),
                        'product_display_name': result_row[1],
                        'article_type': result_row[2],
                        'master_category': result_row[3],
                        'price': float(result_row[4]),
                        'product_image_path': result_row[5],
                        'similarity_score': float(result_row[-1]) if len(result_row) > 6 else 0.0,
                        'rank': rank
                    })
        except Exception as e:
            print(f"\n   ⚠️ Error mapping {item_uid}: {e}")
            continue

    print(f"✅ ({len(all_mappings):,} mappings)")

    # Small delay between batches
    if batch_idx < total_batches - 1:
        time.sleep(0.2)

print(f"\n✅ Mapping complete!")
print(f"   Total mappings: {len(all_mappings):,}")
print(f"   Expected: {df2_count * TOP_K:,}")
print(f"   Success rate: {len(all_mappings) / (df2_count * TOP_K) * 100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Save to Delta Table

# COMMAND ----------

# Convert to DataFrame
mappings_df = spark.createDataFrame(all_mappings)

# Create/replace table
table_name = "main.fashion_demo.df2_to_product_mappings_test" if TEST_MODE else "main.fashion_demo.df2_to_product_mappings"

mappings_df.write.mode("overwrite").saveAsTable(table_name)

saved_count = spark.table(table_name).count()
print(f"✅ Saved {saved_count:,} mappings to {table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Mapping Statistics

# COMMAND ----------

from pyspark.sql.functions import count, avg, min, max, countDistinct

stats = spark.sql(f"""
    SELECT
        COUNT(DISTINCT df2_item_uid) as unique_df2_items,
        COUNT(*) as total_mappings,
        ROUND(AVG(similarity_score), 4) as avg_similarity,
        ROUND(MIN(similarity_score), 4) as min_similarity,
        ROUND(MAX(similarity_score), 4) as max_similarity,
        COUNT(DISTINCT product_id) as unique_products_mapped
    FROM {table_name}
""").collect()[0]

print("MAPPING STATISTICS")
print("="*70)
print(f"Unique DF2 items mapped:     {stats.unique_df2_items:>10,}")
print(f"Total mappings created:      {stats.total_mappings:>10,}")
print(f"Unique products in results:  {stats.unique_products_mapped:>10,}")
print(f"Avg mappings per DF2 item:   {stats.total_mappings / stats.unique_df2_items:>10.1f}")
print(f"\nSimilarity Scores:")
print(f"  Average:                   {stats.avg_similarity:>10.4f}")
print(f"  Minimum:                   {stats.min_similarity:>10.4f}")
print(f"  Maximum:                   {stats.max_similarity:>10.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Similarity Score Distribution

# COMMAND ----------

score_dist = spark.sql(f"""
    SELECT
        CASE
            WHEN similarity_score >= 0.9 THEN '0.9-1.0 (Excellent)'
            WHEN similarity_score >= 0.8 THEN '0.8-0.9 (Very Good)'
            WHEN similarity_score >= 0.7 THEN '0.7-0.8 (Good)'
            WHEN similarity_score >= 0.6 THEN '0.6-0.7 (Fair)'
            ELSE '<0.6 (Poor)'
        END as score_range,
        COUNT(*) as count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as percentage
    FROM {table_name}
    GROUP BY score_range
    ORDER BY score_range DESC
""")

print("\nSIMILARITY SCORE DISTRIBUTION")
print("="*70)
display(score_dist)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Category Distribution

# COMMAND ----------

category_dist = spark.sql(f"""
    SELECT
        master_category,
        COUNT(*) as mapping_count,
        COUNT(DISTINCT product_id) as unique_products,
        ROUND(AVG(similarity_score), 4) as avg_similarity,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as pct_of_total
    FROM {table_name}
    GROUP BY master_category
    ORDER BY mapping_count DESC
""")

print("\nCATEGORY DISTRIBUTION")
print("="*70)
display(category_dist)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 Top Mapped Products

# COMMAND ----------

top_products = spark.sql(f"""
    SELECT
        product_id,
        product_display_name,
        article_type,
        COUNT(*) as times_mapped,
        ROUND(AVG(similarity_score), 4) as avg_score,
        ROUND(AVG(rank), 2) as avg_rank
    FROM {table_name}
    GROUP BY product_id, product_display_name, article_type
    ORDER BY times_mapped DESC
    LIMIT 20
""")

print("\nTOP 20 MOST FREQUENTLY MAPPED PRODUCTS")
print("="*70)
display(top_products)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.5 Rank Distribution

# COMMAND ----------

rank_dist = spark.sql(f"""
    SELECT
        rank,
        COUNT(*) as count,
        ROUND(AVG(similarity_score), 4) as avg_similarity,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as percentage
    FROM {table_name}
    GROUP BY rank
    ORDER BY rank
""")

print("\nRANK DISTRIBUTION (Which ranks get selected?)")
print("="*70)
display(rank_dist)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Visual Exploration 🎨

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Image Preview Setup

# COMMAND ----------

from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_image_from_path(path):
    """Load image from DBFS path"""
    try:
        # Read from DBFS
        with open(path.replace('/Volumes', '/dbfs/Volumes'), 'rb') as f:
            img = Image.open(f)
            return img
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def display_mapping_with_images(df2_uid, num_results=5):
    """Display DF2 item and its top-K mapped products with images"""

    # Get mapping results
    mappings = spark.sql(f"""
        SELECT
            m.df2_item_uid,
            m.product_id,
            m.product_display_name,
            m.article_type,
            m.similarity_score,
            m.rank,
            m.product_image_path
        FROM {table_name} m
        WHERE m.df2_item_uid = '{df2_uid}'
        ORDER BY m.rank
        LIMIT {num_results}
    """).toPandas()

    if len(mappings) == 0:
        print(f"No mappings found for {df2_uid}")
        return

    # Get DF2 item image path
    df2_info = spark.sql(f"""
        SELECT item_uid, filename, image_path
        FROM main.fashion_demo.df2_working_set
        WHERE item_uid = '{df2_uid}'
    """).collect()

    if not df2_info:
        print(f"DF2 item {df2_uid} not found")
        return

    df2_image_path = df2_info[0]['image_path'] if df2_info[0]['image_path'] else None

    # Create subplot
    fig, axes = plt.subplots(1, num_results + 1, figsize=(20, 4))
    fig.suptitle(f'DeepFashion2 Item: {df2_uid} → Top {num_results} Mapped Products',
                 fontsize=14, fontweight='bold')

    # Display DF2 source image
    axes[0].set_title(f'SOURCE\nDF2: {df2_uid}', fontsize=10, fontweight='bold', color='blue')
    if df2_image_path:
        try:
            img = load_image_from_path(df2_image_path)
            if img:
                axes[0].imshow(img)
        except:
            axes[0].text(0.5, 0.5, 'Image\nNot Available',
                        ha='center', va='center', fontsize=10)
    else:
        axes[0].text(0.5, 0.5, 'Image\nNot Available',
                    ha='center', va='center', fontsize=10)
    axes[0].axis('off')

    # Display mapped products
    for idx, row in mappings.iterrows():
        ax = axes[idx + 1]

        # Title with rank and similarity
        title = f"#{row['rank']} (sim: {row['similarity_score']:.3f})\n"
        title += f"{row['product_display_name'][:30]}\n"
        title += f"{row['article_type']}"
        ax.set_title(title, fontsize=8)

        # Load and display image
        if row['product_image_path']:
            try:
                img = load_image_from_path(row['product_image_path'])
                if img:
                    ax.imshow(img)
            except:
                ax.text(0.5, 0.5, 'Image\nNot Available',
                       ha='center', va='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Image\nNot Available',
                   ha='center', va='center', fontsize=8)

        ax.axis('off')

    plt.tight_layout()
    plt.show()

print("✅ Image preview function ready!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Preview Top Mappings

# COMMAND ----------

# Get a few sample DF2 items with their mappings
sample_items = spark.sql(f"""
    SELECT df2_item_uid, COUNT(*) as mapping_count, AVG(similarity_score) as avg_sim
    FROM {table_name}
    GROUP BY df2_item_uid
    ORDER BY avg_sim DESC
    LIMIT 5
""").toPandas()

print("SAMPLE MAPPINGS TO PREVIEW")
print("="*70)
print(sample_items)
print("\nDisplaying first sample...")

# Display first sample
if len(sample_items) > 0:
    first_uid = sample_items.iloc[0]['df2_item_uid']
    display_mapping_with_images(first_uid, num_results=5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3 Interactive Preview
# MAGIC
# MAGIC **Change the `df2_uid` below to preview different mappings**

# COMMAND ----------

# INTERACTIVE: Change this UID to explore different mappings
df2_uid_to_preview = sample_items.iloc[1]['df2_item_uid'] if len(sample_items) > 1 else "35531"

display_mapping_with_images(df2_uid_to_preview, num_results=5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Quality Checks

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1 Check for Missing Mappings

# COMMAND ----------

missing_check = spark.sql(f"""
    SELECT
        d.item_uid,
        d.filename
    FROM main.fashion_demo.df2_working_set d
    LEFT JOIN {table_name} m ON d.item_uid = m.df2_item_uid
    WHERE m.df2_item_uid IS NULL
    {'LIMIT 100' if not TEST_MODE else ''}
""")

missing_count = missing_check.count()

print(f"MISSING MAPPINGS CHECK")
print("="*70)
print(f"DF2 items without mappings: {missing_count}")

if missing_count > 0:
    print("\nSample missing items:")
    display(missing_check.limit(10))
else:
    print("✅ All items have mappings!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.2 Diversity Check

# COMMAND ----------

diversity = spark.sql(f"""
    WITH product_counts AS (
        SELECT
            df2_item_uid,
            COUNT(DISTINCT product_id) as unique_products,
            COUNT(DISTINCT article_type) as unique_article_types,
            COUNT(DISTINCT master_category) as unique_categories
        FROM {table_name}
        GROUP BY df2_item_uid
    )
    SELECT
        ROUND(AVG(unique_products), 2) as avg_unique_products,
        ROUND(AVG(unique_article_types), 2) as avg_unique_article_types,
        ROUND(AVG(unique_categories), 2) as avg_unique_categories,
        MIN(unique_products) as min_unique_products,
        MAX(unique_products) as max_unique_products
    FROM product_counts
""").collect()[0]

print("DIVERSITY METRICS (per DF2 item)")
print("="*70)
print(f"Avg unique products:        {diversity.avg_unique_products:>10.2f} / {TOP_K}")
print(f"Avg unique article types:   {diversity.avg_unique_article_types:>10.2f}")
print(f"Avg unique categories:      {diversity.avg_unique_categories:>10.2f}")
print(f"Min unique products:        {diversity.min_unique_products:>10}")
print(f"Max unique products:        {diversity.max_unique_products:>10}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary & Next Steps

# COMMAND ----------

print("="*70)
print("PHASE 2 MAPPING - SUMMARY")
print("="*70)
print(f"\n{'TEST MODE' if TEST_MODE else 'FULL RUN'}")
print(f"  DF2 items processed:    {df2_count:>10,}")
print(f"  Total mappings:         {len(all_mappings):>10,}")
print(f"  Mappings per item:      {len(all_mappings) / df2_count:>10.1f}")
print(f"  Avg similarity score:   {stats.avg_similarity:>10.4f}")
print(f"\n✅ Table created: {table_name}")
print(f"✅ Mappings saved: {saved_count:,}")

if TEST_MODE:
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review the quality metrics above")
    print("2. Check the image previews to validate mappings")
    print("3. If satisfied, set TEST_MODE = False and run full mapping")
    print("4. Full run will process all 22,000 items (~30 minutes)")
else:
    print("\n" + "="*70)
    print("✅ PHASE 2 COMPLETE - Ready for Phase 3 (Graph Construction)")
    print("="*70)
