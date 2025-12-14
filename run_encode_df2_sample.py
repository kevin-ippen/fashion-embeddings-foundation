#!/usr/bin/env python3
"""
Encode 100 DeepFashion2 images with CLIP to verify embeddings
"""

from databricks.sdk import WorkspaceClient
import requests
import base64
import json
import time
from pathlib import Path

print("="*80)
print("ENCODING 100 DEEPFASHION2 IMAGES WITH CLIP")
print("="*80)
print()

# Initialize
w = WorkspaceClient(profile="DEFAULT")
print(f"✅ Connected to workspace: {w.config.host}\n")

# Get warehouse
warehouses = list(w.warehouses.list())
warehouse = next((wh for wh in warehouses if wh.state.value == "RUNNING"), None)
if not warehouse:
    warehouse = warehouses[0]
warehouse_id = warehouse.id
print(f"✅ Using warehouse: {warehouse.name}\n")

def execute_sql(query, description=None):
    """Execute SQL query"""
    if description:
        print(f"⏳ {description}")
    response = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=query,
        wait_timeout="50s"
    )
    if response.status.state.value == "SUCCEEDED":
        if description:
            print(f"✅ {description}")
        return response.result
    else:
        print(f"❌ Query failed: {response.status.state}")
        return None

# Step 1: Get 100 images from DeepFashion2 volume
print("="*80)
print("STEP 1: SELECTING 100 IMAGES FROM VOLUME")
print("="*80)

result = execute_sql("""
    SELECT
        filename as image_name,
        image_path
    FROM main.fashion_demo.df2_working_set
    WHERE filename IS NOT NULL
      AND image_path IS NOT NULL
    ORDER BY RAND()
    LIMIT 100
""", "Selecting 100 random images from working set")

if not result or not result.data_array:
    print("❌ Could not get images")
    exit(1)

images = [(row[0], row[1]) for row in result.data_array]
print(f"\n✅ Selected {len(images)} images")
print(f"Sample: {images[0][0]}, {images[1][0]}, {images[2][0]}...\n")

# Step 2: Get CLIP endpoint info
print("="*80)
print("STEP 2: CONNECTING TO CLIP ENDPOINT")
print("="*80)

endpoint_name = "clip-image-encoder"
try:
    endpoint = w.serving_endpoints.get(endpoint_name)
    print(f"✅ Found endpoint: {endpoint_name}")
    print(f"   State: {endpoint.state.ready}")

    # Get endpoint URL
    endpoint_url = f"{w.config.host}/serving-endpoints/{endpoint_name}/invocations"
    print(f"   URL: {endpoint_url}\n")
except Exception as e:
    print(f"❌ Error getting endpoint: {e}")
    exit(1)

# Step 3: Encode images in batches
print("="*80)
print("STEP 3: ENCODING IMAGES WITH CLIP")
print("="*80)

BATCH_SIZE = 10
embeddings_data = []
errors = 0

headers = {
    "Authorization": f"Bearer {w.config.token}",
    "Content-Type": "application/json"
}

total_batches = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(total_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(images))
    batch = images[start_idx:end_idx]

    print(f"\nBatch {batch_idx + 1}/{total_batches}: Processing images {start_idx + 1}-{end_idx}...")

    for image_name, image_path in batch:
        try:
            # Read image file from Volume (using dbutils or requests)
            # For now, let's use the Databricks Files API
            file_url = f"{w.config.host}/api/2.0/fs/files{image_path}"
            file_response = requests.get(file_url, headers={"Authorization": f"Bearer {w.config.token}"})

            if file_response.status_code != 200:
                print(f"  ⚠️  Could not read {image_name}: {file_response.status_code}")
                errors += 1
                continue

            # Encode image as base64
            image_bytes = file_response.content
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            # Call CLIP endpoint
            payload = {
                "dataframe_records": [{
                    "image": image_b64
                }]
            }

            response = requests.post(endpoint_url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result_data = response.json()
                # Extract embedding from response
                if 'predictions' in result_data and len(result_data['predictions']) > 0:
                    embedding = result_data['predictions'][0]
                    embeddings_data.append({
                        'image_name': image_name,
                        'image_path': image_path,
                        'embedding': embedding
                    })
                    print(f"  ✅ {image_name} -> {len(embedding)}D embedding")
                else:
                    print(f"  ⚠️  {image_name}: No predictions in response")
                    errors += 1
            else:
                print(f"  ⚠️  {image_name}: Endpoint error {response.status_code}")
                errors += 1

        except Exception as e:
            print(f"  ⚠️  {image_name}: Error - {e}")
            errors += 1
            continue

    # Rate limiting
    if batch_idx < total_batches - 1:
        time.sleep(0.5)

print(f"\n✅ Encoding complete:")
print(f"   Success: {len(embeddings_data)}/{len(images)}")
print(f"   Errors: {errors}\n")

if len(embeddings_data) == 0:
    print("❌ No embeddings generated")
    exit(1)

# Step 4: Store embeddings in Delta table
print("="*80)
print("STEP 4: STORING EMBEDDINGS IN DELTA TABLE")
print("="*80)

# Create test table
execute_sql("""
    CREATE OR REPLACE TABLE main.fashion_demo.df2_clip_test_100 (
        image_name STRING,
        image_path STRING,
        clip_embedding ARRAY<DOUBLE>,
        created_at TIMESTAMP
    ) USING DELTA
""", "Creating test table")

# Insert in batches
INSERT_BATCH = 20
total_inserts = (len(embeddings_data) + INSERT_BATCH - 1) // INSERT_BATCH

for batch_idx in range(total_inserts):
    start = batch_idx * INSERT_BATCH
    end = min(start + INSERT_BATCH, len(embeddings_data))
    batch = embeddings_data[start:end]

    values = []
    for item in batch:
        name_esc = item['image_name'].replace("'", "''")
        path_esc = item['image_path'].replace("'", "''")
        emb_str = '[' + ','.join([str(float(x)) for x in item['embedding']]) + ']'
        values.append(f"('{name_esc}', '{path_esc}', {emb_str}, current_timestamp())")

    execute_sql(f"""
        INSERT INTO main.fashion_demo.df2_clip_test_100
        VALUES {', '.join(values)}
    """)

# Verify
result = execute_sql("SELECT COUNT(*) FROM main.fashion_demo.df2_clip_test_100")
count = int(result.data_array[0][0]) if result and result.data_array else 0
print(f"\n✅ Stored {count} embeddings in main.fashion_demo.df2_clip_test_100\n")

# Step 5: Validate embeddings
print("="*80)
print("STEP 5: VALIDATING EMBEDDINGS")
print("="*80)

result = execute_sql("""
    SELECT
        COUNT(*) as total,
        COUNT(clip_embedding) as has_embedding,
        COUNT(CASE WHEN SIZE(clip_embedding) = 512 THEN 1 END) as valid_512d
    FROM main.fashion_demo.df2_clip_test_100
""", "Checking embedding dimensions")

if result and result.data_array:
    stats = result.data_array[0]
    print(f"\nValidation Results:")
    print(f"  Total rows:        {int(stats[0]):>6}")
    print(f"  Has embeddings:    {int(stats[1]):>6}")
    print(f"  Valid 512D:        {int(stats[2]):>6}")

# Check diversity
result = execute_sql("""
    SELECT clip_embedding
    FROM main.fashion_demo.df2_clip_test_100
    LIMIT 10
""", "Sampling embeddings for diversity check")

if result and result.data_array:
    import numpy as np

    embeddings = []
    for row in result.data_array:
        emb_str = row[0]
        if isinstance(emb_str, str):
            emb_list = json.loads(emb_str)
            embeddings.append(np.array(emb_list, dtype=np.float32))
        elif isinstance(emb_str, (list, tuple)):
            embeddings.append(np.array(emb_str, dtype=np.float32))

    if len(embeddings) >= 2:
        first = embeddings[0]
        similarities = [np.dot(first, emb) for emb in embeddings[1:]]

        print(f"\nDiversity Check (9 comparisons):")
        print(f"  Mean similarity:   {np.mean(similarities):.4f}")
        print(f"  Std deviation:     {np.std(similarities):.4f}")
        print(f"  Range:             [{np.min(similarities):.4f}, {np.max(similarities):.4f}]")

        if np.std(similarities) > 0.05:
            print(f"\n✅ Embeddings are diverse!")
        else:
            print(f"\n⚠️  Low diversity detected")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
✅ Successfully encoded {len(embeddings_data)} DeepFashion2 images

Endpoint: {endpoint_name}
Table: main.fashion_demo.df2_clip_test_100

Next Steps:
1. Compare these embeddings with existing df2_working_set embeddings
2. If quality is good, scale up to encode all ~10K images
3. Replace df2_working_set with fresh embeddings
""")
