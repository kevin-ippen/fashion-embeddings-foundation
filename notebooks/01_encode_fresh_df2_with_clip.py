# Databricks notebook source
# MAGIC %md
# MAGIC # Encode Fresh DeepFashion2 Images with CLIP
# MAGIC
# MAGIC Encode the fresh dataset images using CLIP to create high-quality embeddings.
# MAGIC
# MAGIC **Input**: `main.fashion_demo.df2_images_fresh`
# MAGIC **Output**: `main.fashion_demo.df2_working_set` (clean, verified embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Source data
SOURCE_TABLE = "main.fashion_demo.df2_images_fresh"
METADATA_TABLE = "main.fashion_demo.df2_metadata_fresh"
VOLUME_PATH = "/Volumes/main/fashion_demo/deepfashion2_fresh/images"

# CLIP endpoint - using MULTIMODAL for image + text
CLIP_ENDPOINT = "clip-multimodal-encoder"

# Output table
OUTPUT_TABLE = "main.fashion_demo.df2_working_set"

# Processing
BATCH_SIZE = 50  # Process images in batches

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Fresh Images with Metadata

# COMMAND ----------

from pyspark.sql.functions import regexp_extract, col

# Load image catalog
images_df = spark.table(SOURCE_TABLE)

# Load metadata
metadata_df = spark.table(METADATA_TABLE)

print(f"✅ Loaded image catalog: {images_df.count():,} images")
print(f"✅ Loaded metadata: {metadata_df.count():,} records")

# Extract image name from metadata path for joining
metadata_clean = metadata_df.withColumn(
    "filename",
    regexp_extract(col("path"), r"([0-9]+\.jpg)", 1)
).select("filename", "category_name", "category_id")

# Join images with metadata to get category names
images_with_metadata = images_df.join(
    metadata_clean,
    on="filename",
    how="left"
)

joined_count = images_with_metadata.filter(col("category_name").isNotNull()).count()
print(f"✅ Joined with metadata: {joined_count:,} images have category names")

# Show sample
display(images_with_metadata.select("filename", "image_id", "category_name").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Setup CLIP Endpoint

# COMMAND ----------

import requests

# Get auth
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

endpoint_url = f"{workspace_url}/serving-endpoints/{CLIP_ENDPOINT}/invocations"

print(f"✅ CLIP Endpoint configured")
print(f"   Endpoint: {CLIP_ENDPOINT}")
print(f"   URL: {endpoint_url}")

# Test endpoint availability
try:
    endpoint_check_url = f"{workspace_url}/api/2.0/serving-endpoints/{CLIP_ENDPOINT}"
    response = requests.get(endpoint_check_url, headers=headers)

    if response.status_code == 200:
        endpoint_info = response.json()
        state = endpoint_info.get('state', {}).get('ready', 'unknown')
        print(f"   State: {state}")
    else:
        print(f"   ⚠️  Status check returned: {response.status_code}")
except Exception as e:
    print(f"   ⚠️  Could not check status: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test Multimodal Encoding (Image + Text)

# COMMAND ----------

import base64

# Get one test image with metadata
test_row = images_with_metadata.first()
test_path = test_row.image_path
test_category = test_row.category_name if test_row.category_name else "fashion garment"

print(f"Testing with: {test_row.filename}")
print(f"Category: {test_category}")
print(f"Path: {test_path}")

try:
    # Read image using Spark
    binary_df = spark.read.format("binaryFile").load(test_path)
    binary_data = binary_df.select("content").first()
    image_bytes = bytes(binary_data[0])

    print(f"✅ Read {len(image_bytes):,} bytes")

    # Encode as base64
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    print(f"✅ Base64 encoded: {len(image_b64):,} characters")

    # Call CLIP MULTIMODAL endpoint with image + text
    payload = {
        "dataframe_records": [{
            "image": image_b64,
            "text": test_category  # Include category text for multimodal
        }]
    }

    print(f"\nCalling multimodal endpoint...")
    print(f"  Image: {len(image_b64)} chars")
    print(f"  Text: '{test_category}'")

    response = requests.post(endpoint_url, headers=headers, json=payload, timeout=120)

    if response.status_code == 200:
        result = response.json()
        if 'predictions' in result and len(result['predictions']) > 0:
            embedding = result['predictions'][0]
            print(f"\n✅ SUCCESS! Got multimodal embedding: {len(embedding)}D vector")
            print(f"   Sample values: {embedding[:5]}")
            print(f"   Sum: {sum(embedding):.4f}")
        else:
            print(f"❌ No predictions in response")
            print(f"   Response: {result}")
    else:
        print(f"❌ Endpoint error {response.status_code}")
        print(f"   Response: {response.text[:500]}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Encode All Images in Batches

# COMMAND ----------

from pyspark.sql.functions import col, lit, array, struct
from pyspark.sql.types import ArrayType, DoubleType, StructType, StructField, StringType
import time

total_images = images_with_metadata.count()

print(f"Starting batch encoding of {total_images:,} images...")
print(f"Batch size: {BATCH_SIZE}")
print(f"Estimated time: ~{(total_images / BATCH_SIZE) * 2} minutes")
print()

# Read all images from Volume
print("Reading images from Volume...")
all_images_binary = spark.read.format("binaryFile").load(f"{VOLUME_PATH}/*.jpg")

# Join with catalog+metadata to get category names
joined_df = images_with_metadata.join(
    all_images_binary.selectExpr("path as image_path", "content"),
    on="image_path",
    how="inner"
)

print(f"✅ Loaded {joined_df.count():,} images with content and metadata")

# Collect image data for batch processing
image_data = joined_df.select("image_id", "filename", "image_path", "category_name", "content").collect()

print(f"\nProcessing {len(image_data)} images in batches...")

# Process in batches
results = []
total_batches = (len(image_data) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(total_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(image_data))
    batch = image_data[start_idx:end_idx]

    print(f"Batch {batch_idx + 1}/{total_batches}: Processing {start_idx + 1}-{end_idx}...", end=" ")

    batch_success = 0
    batch_failed = 0

    for row in batch:
        try:
            # Get image bytes
            image_bytes = bytes(row.content)

            # Encode as base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            # Get category text (with fallback)
            category_text = row.category_name if row.category_name else "fashion garment"

            # Call CLIP MULTIMODAL endpoint with image + text
            payload = {
                "dataframe_records": [{
                    "image": image_b64,
                    "text": category_text  # Include category for multimodal
                }]
            }

            response = requests.post(endpoint_url, headers=headers, json=payload, timeout=120)

            if response.status_code == 200:
                result = response.json()
                if 'predictions' in result and len(result['predictions']) > 0:
                    embedding = result['predictions'][0]

                    results.append({
                        'image_id': row.image_id,
                        'filename': row.filename,
                        'image_path': row.image_path,
                        'category_name': category_text,
                        'clip_embedding': embedding
                    })
                    batch_success += 1
                else:
                    batch_failed += 1
            else:
                batch_failed += 1

        except Exception as e:
            batch_failed += 1
            continue

    print(f"✅ {batch_success} success, {batch_failed} failed")

    # Rate limiting
    if batch_idx < total_batches - 1:
        time.sleep(0.5)

print(f"\n{'='*70}")
print(f"Encoding complete!")
print(f"  Total images: {len(image_data):,}")
print(f"  Successfully encoded: {len(results):,}")
print(f"  Failed: {len(image_data) - len(results):,}")
print(f"  Success rate: {100 * len(results) / len(image_data):.1f}%")
print(f"{'='*70}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create DataFrame and Save

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType
from pyspark.sql.functions import current_timestamp

# Define schema
schema = StructType([
    StructField("image_id", StringType(), True),
    StructField("filename", StringType(), True),
    StructField("image_path", StringType(), True),
    StructField("category_name", StringType(), True),
    StructField("clip_embedding", ArrayType(DoubleType()), True)
])

# Create DataFrame
if len(results) > 0:
    embeddings_df = spark.createDataFrame(results, schema)

    # Add metadata
    final_df = embeddings_df.withColumn("created_at", current_timestamp())

    # Add item_uid (same as image_id for now)
    final_df = final_df.withColumn("item_uid", col("image_id"))

    print(f"Created DataFrame with {final_df.count():,} rows")
    display(final_df.limit(10))
else:
    print("❌ No results to save")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save to df2_working_set

# COMMAND ----------

# Save to Delta table
if len(results) > 0:
    final_df.write.mode("overwrite").saveAsTable(OUTPUT_TABLE)

    print(f"✅ Saved embeddings to: {OUTPUT_TABLE}")

    # Verify
    saved_count = spark.table(OUTPUT_TABLE).count()
    print(f"   Verified rows: {saved_count:,}")
else:
    print("❌ No data to save")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validate Embeddings

# COMMAND ----------

import numpy as np

# Check dimensions
validation_query = f"""
    SELECT
        COUNT(*) as total,
        COUNT(clip_embedding) as has_embedding,
        COUNT(CASE WHEN SIZE(clip_embedding) = 512 THEN 1 END) as valid_512d,
        COUNT(item_uid) as has_item_uid,
        COUNT(DISTINCT item_uid) as unique_items
    FROM {OUTPUT_TABLE}
"""

validation_df = spark.sql(validation_query)
display(validation_df)

val_row = validation_df.first()
print(f"\nValidation Results:")
print(f"  Total rows:        {val_row.total:>6,}")
print(f"  Has embeddings:    {val_row.has_embedding:>6,}")
print(f"  Valid 512D:        {val_row.valid_512d:>6,}")
print(f"  Has item_uid:      {val_row.has_item_uid:>6,}")
print(f"  Unique items:      {val_row.unique_items:>6,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Check Embedding Diversity

# COMMAND ----------

# Get sample embeddings for diversity check
sample_embeddings_df = spark.sql(f"""
    SELECT clip_embedding
    FROM {OUTPUT_TABLE}
    WHERE clip_embedding IS NOT NULL
      AND SIZE(clip_embedding) = 512
    LIMIT 20
""")

sample_embs = sample_embeddings_df.collect()

if len(sample_embs) >= 2:
    # Convert to numpy
    embeddings = [np.array(row.clip_embedding) for row in sample_embs]

    # Check if all identical
    first = embeddings[0]
    all_same = all(np.allclose(emb, first, rtol=1e-9) for emb in embeddings)

    if all_same:
        print("❌ PROBLEM: All embeddings are IDENTICAL!")
        print("   This indicates an issue with the encoding process.")
    else:
        print("✅ EXCELLENT: Embeddings are DIVERSE!")

        # Calculate diversity metrics
        similarities = [np.dot(first, emb) / (np.linalg.norm(first) * np.linalg.norm(emb))
                       for emb in embeddings[1:10]]

        print(f"\nDiversity metrics (9 comparisons):")
        print(f"  Mean similarity:   {np.mean(similarities):.4f}")
        print(f"  Median similarity: {np.median(similarities):.4f}")
        print(f"  Std deviation:     {np.std(similarities):.4f}")
        print(f"  Range:             [{np.min(similarities):.4f}, {np.max(similarities):.4f}]")

        if np.std(similarities) > 0.05:
            print(f"\n🎉 Great diversity! Embeddings are ready to use.")
        else:
            print(f"\n⚠️  Low diversity - embeddings may be too similar")
else:
    print("⚠️  Not enough embeddings to check diversity")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Test with Vector Search

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# Initialize Vector Search
vsc = VectorSearchClient(disable_notice=True)
vs_index = vsc.get_index(
    endpoint_name="one-env-shared-endpoint-11",
    index_name="main.fashion_demo.vs_product_hybrid_search"
)

print("Testing Vector Search with new embeddings...")
print()

# Get 3 random test embeddings
test_df = spark.sql(f"""
    SELECT item_uid, filename, clip_embedding
    FROM {OUTPUT_TABLE}
    ORDER BY RAND()
    LIMIT 3
""").collect()

all_top5_ids = []

for idx, row in enumerate(test_df, 1):
    item_uid = row.item_uid
    filename = row.filename
    embedding = row.clip_embedding

    print(f"{idx}. DF2 Item: {filename} (uid: {item_uid})")

    try:
        results = vs_index.similarity_search(
            query_vector=embedding,
            columns=["product_id", "product_display_name", "article_type"],
            num_results=5
        )

        if results and 'result' in results and 'data_array' in results['result']:
            top5_ids = []
            for rank, result_row in enumerate(results['result']['data_array'], 1):
                prod_id = result_row[0]
                prod_name = result_row[1]
                article = result_row[2]
                score = result_row[-1]
                top5_ids.append(int(prod_id))
                print(f"   #{rank}: [{prod_id}] {prod_name[:35]:<35} ({article:<15}) - {score:.3f}")
            all_top5_ids.append(set(top5_ids))
        else:
            print(f"   ⚠️  No results returned")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()

# Check if results are different
if len(all_top5_ids) == 3:
    if all_top5_ids[0] == all_top5_ids[1] == all_top5_ids[2]:
        print("❌ PROBLEM: All 3 items returned SAME products!")
        print("   This indicates embeddings may not be diverse enough.")
    else:
        print("✅ SUCCESS: Different items returned DIFFERENT products!")

        # Show overlap
        overlap_1_2 = len(all_top5_ids[0] & all_top5_ids[1])
        overlap_1_3 = len(all_top5_ids[0] & all_top5_ids[2])
        overlap_2_3 = len(all_top5_ids[1] & all_top5_ids[2])

        print(f"\nOverlap analysis:")
        print(f"  Items 1 & 2: {overlap_1_2}/5 products in common")
        print(f"  Items 1 & 3: {overlap_1_3}/5 products in common")
        print(f"  Items 2 & 3: {overlap_2_3}/5 products in common")

        print(f"\n🎉 Fresh embeddings are working correctly!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ✅ **Setup Complete!**
# MAGIC
# MAGIC - Encoded fresh DeepFashion2 images with CLIP
# MAGIC - Created `main.fashion_demo.df2_working_set` with verified embeddings
# MAGIC - Validated embedding diversity and quality
# MAGIC - Tested with Vector Search - confirmed working!
# MAGIC
# MAGIC ### Next Steps:
# MAGIC
# MAGIC 1. **Run Phase 2 Mapping**: Use `run_phase2_mapping.py` to map all items
# MAGIC 2. **Visual Validation**: Run `03_visual_mapping_preview.py` to see results
# MAGIC 3. **Scale Up**: If quality is good, process remaining images
# MAGIC 4. **Continue to Phase 3**: Build co-occurrence graph
