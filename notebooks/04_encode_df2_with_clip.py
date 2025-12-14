# Databricks notebook source
# MAGIC %md
# MAGIC # Re-encode DeepFashion2 Images with CLIP
# MAGIC
# MAGIC Test encoding 100 DeepFashion2 images using the clip-image-encoder endpoint to verify embedding quality.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Number of images to test
NUM_IMAGES = 100

# CLIP endpoint
CLIP_ENDPOINT = "clip-image-encoder"

# Source and destination tables
SOURCE_TABLE = "main.fashion_demo.df2_working_set"
OUTPUT_TABLE = "main.fashion_demo.df2_clip_reencoded_100"

# Volume path
VOLUME_BASE = "/Volumes/main/fashion_demo/deepfashion2/fashion-dataset/fashion-dataset/images"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Select Sample Images

# COMMAND ----------

# Get 100 random images that we know exist
sample_df = spark.sql(f"""
    SELECT
        item_uid,
        filename,
        image_path,
        clip_embedding as original_embedding
    FROM {SOURCE_TABLE}
    WHERE filename IS NOT NULL
    ORDER BY RAND()
    LIMIT {NUM_IMAGES}
""")

print(f"Selected {sample_df.count()} images for re-encoding")
display(sample_df.select("item_uid", "filename").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Test Reading One Image

# COMMAND ----------

# Test reading a single image file
test_row = sample_df.first()
test_filename = test_row.filename
test_path = f"{VOLUME_BASE}/{test_filename}"

print(f"Test image: {test_filename}")
print(f"Full path: {test_path}")

# Read image using dbutils
try:
    import base64
    with open(test_path.replace("/Volumes", "/dbfs/Volumes"), "rb") as f:
        image_bytes = f.read()
    print(f"✅ Successfully read {len(image_bytes)} bytes")

    # Test base64 encoding
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    print(f"✅ Base64 encoded: {len(image_b64)} characters")
except Exception as e:
    print(f"❌ Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test CLIP Endpoint with One Image

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

# Get endpoint
client = MlflowClient()
endpoint_url = f"databricks-model-serving://{CLIP_ENDPOINT}"

print(f"Testing CLIP endpoint: {CLIP_ENDPOINT}")

try:
    # Prepare payload
    payload = {
        "dataframe_records": [{
            "image": image_b64
        }]
    }

    # Call endpoint using mlflow
    import requests
    import os

    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        f"{workspace_url}/serving-endpoints/{CLIP_ENDPOINT}/invocations",
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code == 200:
        result = response.json()
        if 'predictions' in result and len(result['predictions']) > 0:
            embedding = result['predictions'][0]
            print(f"✅ Got embedding: {len(embedding)}D vector")
            print(f"   Sample values: {embedding[:5]}")
        else:
            print(f"❌ No predictions in response: {result}")
    else:
        print(f"❌ Endpoint error {response.status_code}: {response.text}")

except Exception as e:
    print(f"❌ Error calling endpoint: {e}")
    import traceback
    traceback.print_exc()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Encode All 100 Images

# COMMAND ----------

import base64
import requests
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, DoubleType, StringType
import time

# Get auth
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

endpoint_url = f"{workspace_url}/serving-endpoints/{CLIP_ENDPOINT}/invocations"

def encode_image_clip(filename):
    """Encode a single image using CLIP endpoint"""
    if not filename:
        return None

    try:
        # Read image from Volume
        image_path = f"/dbfs/Volumes/main/fashion_demo/deepfashion2/fashion-dataset/fashion-dataset/images/{filename}"

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Encode as base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # Call CLIP endpoint
        payload = {
            "dataframe_records": [{
                "image": image_b64
            }]
        }

        response = requests.post(endpoint_url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            if 'predictions' in result and len(result['predictions']) > 0:
                return result['predictions'][0]

        return None

    except Exception as e:
        print(f"Error encoding {filename}: {e}")
        return None

# Register UDF
encode_udf = udf(encode_image_clip, ArrayType(DoubleType()))

# Encode images (this will take a while - ~100 images * ~1-2 seconds each)
print(f"Encoding {NUM_IMAGES} images with CLIP...")
print("This will take 2-5 minutes...")

encoded_df = sample_df.withColumn("new_embedding", encode_udf(col("filename")))

# Filter out failed encodings
success_df = encoded_df.filter(col("new_embedding").isNotNull())

success_count = success_df.count()
print(f"\n✅ Successfully encoded: {success_count}/{NUM_IMAGES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Save to Delta Table

# COMMAND ----------

from pyspark.sql.functions import current_timestamp

# Save results
result_df = success_df.select(
    col("item_uid"),
    col("filename"),
    col("image_path"),
    col("original_embedding"),
    col("new_embedding"),
    current_timestamp().alias("created_at")
)

result_df.write.mode("overwrite").saveAsTable(OUTPUT_TABLE)

print(f"✅ Saved {success_count} re-encoded embeddings to {OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Validate Embeddings

# COMMAND ----------

# Check dimensions
validation_df = spark.sql(f"""
    SELECT
        COUNT(*) as total,
        COUNT(new_embedding) as has_new_embedding,
        COUNT(CASE WHEN SIZE(new_embedding) = 512 THEN 1 END) as valid_512d,
        COUNT(original_embedding) as has_original,
        COUNT(CASE WHEN SIZE(original_embedding) = 512 THEN 1 END) as original_512d
    FROM {OUTPUT_TABLE}
""")

display(validation_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Compare Original vs New Embeddings

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

# Get sample for comparison
sample_pd = spark.sql(f"""
    SELECT original_embedding, new_embedding
    FROM {OUTPUT_TABLE}
    WHERE original_embedding IS NOT NULL
      AND new_embedding IS NOT NULL
    LIMIT 10
""").toPandas()

if len(sample_pd) > 0:
    # Calculate similarities between old and new
    similarities = []
    for _, row in sample_pd.iterrows():
        old_emb = np.array(row['original_embedding'])
        new_emb = np.array(row['new_embedding'])

        # Cosine similarity
        sim = np.dot(old_emb, new_emb) / (np.linalg.norm(old_emb) * np.linalg.norm(new_emb))
        similarities.append(sim)

    print(f"Similarity between original and new embeddings:")
    print(f"  Mean:   {np.mean(similarities):.4f}")
    print(f"  Median: {np.median(similarities):.4f}")
    print(f"  Std:    {np.std(similarities):.4f}")
    print(f"  Range:  [{np.min(similarities):.4f}, {np.max(similarities):.4f}]")

    # Check new embedding diversity
    new_embs = [np.array(row['new_embedding']) for _, row in sample_pd.iterrows()]
    first_new = new_embs[0]
    div_sims = [np.dot(first_new, emb) for emb in new_embs[1:]]

    print(f"\nNew embeddings diversity (9 comparisons):")
    print(f"  Mean similarity:   {np.mean(div_sims):.4f}")
    print(f"  Std deviation:     {np.std(div_sims):.4f}")
    print(f"  Range:             [{np.min(div_sims):.4f}, {np.max(div_sims):.4f}]")

    if np.std(div_sims) > 0.05:
        print(f"\n✅ New embeddings are diverse!")
    else:
        print(f"\n⚠️  Low diversity detected")

else:
    print("No data to compare")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Test Vector Search with New Embeddings

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# Initialize Vector Search
vsc = VectorSearchClient(disable_notice=True)
vs_index = vsc.get_index(
    endpoint_name="one-env-shared-endpoint-11",
    index_name="main.fashion_demo.vs_product_hybrid_search"
)

print("Testing Vector Search with new embeddings...")

# Get 3 test embeddings
test_df = spark.sql(f"""
    SELECT item_uid, new_embedding
    FROM {OUTPUT_TABLE}
    LIMIT 3
""").toPandas()

for idx, row in test_df.iterrows():
    item_uid = row['item_uid']
    embedding = row['new_embedding']

    print(f"\n{idx + 1}. Item {item_uid}:")

    try:
        results = vs_index.similarity_search(
            query_vector=embedding,
            columns=["product_id", "product_display_name", "article_type"],
            num_results=5
        )

        if results and 'result' in results and 'data_array' in results['result']:
            for rank, result_row in enumerate(results['result']['data_array'], 1):
                prod_id = result_row[0]
                prod_name = result_row[1]
                article = result_row[2]
                score = result_row[-1]
                print(f"   #{rank}: [{prod_id}] {prod_name[:40]:<40} ({article}) - {score:.3f}")
    except Exception as e:
        print(f"   Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC - Re-encoded 100 DeepFashion2 images using CLIP
# MAGIC - Saved to: `main.fashion_demo.df2_clip_reencoded_100`
# MAGIC - Validated embedding dimensions and diversity
# MAGIC - Tested with Vector Search
# MAGIC
# MAGIC **Next Steps:**
# MAGIC 1. If quality is good, scale up to all ~617 images in df2_working_set
# MAGIC 2. Or scale up to full ~10K DeepFashion2 dataset if images are available in Volume
