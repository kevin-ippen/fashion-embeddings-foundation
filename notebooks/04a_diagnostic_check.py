# Databricks notebook source
# MAGIC %md
# MAGIC # Diagnostic Check for CLIP Encoding
# MAGIC
# MAGIC Quick checks to identify issues before running the full encoding notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 1: Volume Access

# COMMAND ----------

import os

volume_path = "/Volumes/main/fashion_demo/deepfashion2/fashion-dataset/fashion-dataset/images"
dbfs_path = "/dbfs" + volume_path

print(f"Testing Volume access...")
print(f"Volume path: {volume_path}")
print(f"DBFS path:   {dbfs_path}")

# Check if path exists
if os.path.exists(dbfs_path):
    print(f"✅ Volume path exists")

    # List files
    try:
        files = os.listdir(dbfs_path)
        print(f"✅ Can list files: {len(files)} files found")
        print(f"   Sample files: {files[:5]}")
    except Exception as e:
        print(f"❌ Cannot list files: {e}")
else:
    print(f"❌ Volume path does not exist")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 2: Read Test Image

# COMMAND ----------

# Try to read one specific image we know exists using Spark
test_file = "10063.jpg"
test_path = f"{volume_path}/{test_file}"

print(f"Testing image read: {test_file}")
print(f"Using Spark binary file reader...")

try:
    # Read using Spark's binary file format
    binary_df = spark.read.format("binaryFile").load(test_path)
    binary_data = binary_df.select("content").first()
    image_bytes = bytes(binary_data[0])

    print(f"✅ Successfully read {len(image_bytes):,} bytes using Spark")

    # Try base64 encoding
    import base64
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    print(f"✅ Base64 encoded: {len(image_b64):,} characters")

except Exception as e:
    print(f"❌ Error reading file: {e}")
    import traceback
    traceback.print_exc()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 3: Database Access

# COMMAND ----------

# Check if we can query the table
try:
    result = spark.sql("""
        SELECT COUNT(*) as count,
               COUNT(filename) as has_filename
        FROM main.fashion_demo.df2_working_set
    """)

    row = result.first()
    print(f"✅ Can query df2_working_set")
    print(f"   Total rows: {row['count']}")
    print(f"   Has filename: {row['has_filename']}")

    # Get sample filenames
    sample = spark.sql("""
        SELECT filename
        FROM main.fashion_demo.df2_working_set
        WHERE filename IS NOT NULL
        LIMIT 3
    """).toPandas()

    print(f"   Sample filenames: {list(sample['filename'])}")

except Exception as e:
    print(f"❌ Error querying table: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 4: CLIP Endpoint

# COMMAND ----------

import requests

endpoint_name = "clip-image-encoder"

# Get auth token
try:
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

    print(f"✅ Got auth token")
    print(f"   Workspace: {workspace_url}")
    print(f"   Token: {token[:20]}...")

    # Check endpoint exists
    headers = {"Authorization": f"Bearer {token}"}
    endpoint_url = f"{workspace_url}/api/2.0/serving-endpoints/{endpoint_name}"

    response = requests.get(endpoint_url, headers=headers)

    if response.status_code == 200:
        endpoint_info = response.json()
        print(f"✅ Endpoint '{endpoint_name}' exists")
        print(f"   State: {endpoint_info.get('state', {}).get('ready', 'unknown')}")
    else:
        print(f"❌ Endpoint check failed: {response.status_code}")
        print(f"   Response: {response.text}")

except Exception as e:
    print(f"❌ Error checking endpoint: {e}")
    import traceback
    traceback.print_exc()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 5: Test CLIP Invocation

# COMMAND ----------

# Try calling CLIP with test image
if 'image_b64' in dir() and 'token' in dir() and 'workspace_url' in dir():

    print(f"Testing CLIP endpoint invocation...")

    invocation_url = f"{workspace_url}/serving-endpoints/{endpoint_name}/invocations"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "dataframe_records": [{
            "image": image_b64
        }]
    }

    try:
        response = requests.post(invocation_url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ CLIP endpoint working!")

            if 'predictions' in result and len(result['predictions']) > 0:
                embedding = result['predictions'][0]
                print(f"   Got embedding: {len(embedding)}D vector")
                print(f"   Sample values: {embedding[:5]}")
            else:
                print(f"   ⚠️ Response structure: {result.keys()}")

        else:
            print(f"❌ Endpoint returned error {response.status_code}")
            print(f"   Response: {response.text[:500]}")

    except Exception as e:
        print(f"❌ Error calling endpoint: {e}")
        import traceback
        traceback.print_exc()

else:
    print("⚠️ Skipping - previous checks failed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Review the results above. All checks should show ✅ before running the full encoding notebook.
# MAGIC
# MAGIC Common issues:
# MAGIC - **Volume access**: Check Unity Catalog permissions
# MAGIC - **File not found**: Verify image files exist in Volume
# MAGIC - **Endpoint error**: Check if CLIP endpoint is running
# MAGIC - **Auth error**: Verify workspace token permissions
