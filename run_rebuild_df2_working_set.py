#!/usr/bin/env python3
"""
CRITICAL FIX: Rebuild df2_working_set from df2_items

Issue: deepfashion2_clip_embeddings has 22K identical embeddings
Solution: Use df2_items (10,617 rows) with diverse embeddings
"""

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
import numpy as np
import time
import json

print("="*80)
print("REBUILDING df2_working_set FROM df2_items")
print("="*80)
print()

# Initialize
print("Initializing Databricks client...")
w = WorkspaceClient(profile="DEFAULT")
print(f"✅ Connected to workspace: {w.config.host}\n")

# Get SQL warehouse
print("Finding SQL warehouse...")
warehouses = w.warehouses.list()
warehouse = next((wh for wh in warehouses if wh.state.value == "RUNNING"), None)
if not warehouse:
    warehouse = next(iter(warehouses), None)
    if warehouse:
        print(f"⏳ Starting warehouse: {warehouse.name}")
        w.warehouses.start(warehouse.id)
        time.sleep(10)

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
        if response.status.error:
            print(f"   Error: {response.status.error}")
        return None

# Step 1: Validate df2_items source
print("="*80)
print("STEP 1: VALIDATING df2_items SOURCE")
print("="*80)

result = execute_sql("""
    SELECT
        COUNT(*) as total_rows,
        COUNT(clip_embedding) as has_embedding,
        COUNT(CASE WHEN SIZE(clip_embedding) = 512 THEN 1 END) as valid_512d,
        COUNT(image_path) as has_image_path
    FROM main.fashion_demo.df2_items
""", "Checking df2_items statistics")

if result and result.data_array:
    stats = result.data_array[0]
    print(f"\ndf2_items Statistics:")
    print(f"  Total rows:        {int(stats[0]):>10,}")
    print(f"  Has embeddings:    {int(stats[1]):>10,}")
    print(f"  Valid 512D:        {int(stats[2]):>10,}")
    print(f"  Has image paths:   {int(stats[3]):>10,}")

    if int(stats[2]) == 0:
        print("\n❌ ERROR: No valid embeddings in df2_items!")
        exit(1)
else:
    print("❌ Could not read df2_items")
    exit(1)

# Step 2: Check embedding diversity (CRITICAL)
print("\n" + "="*80)
print("STEP 2: CHECKING EMBEDDING DIVERSITY (CRITICAL)")
print("="*80)

result = execute_sql("""
    SELECT clip_embedding
    FROM main.fashion_demo.df2_items
    WHERE clip_embedding IS NOT NULL
      AND SIZE(clip_embedding) = 512
    LIMIT 50
""", "Fetching 50 sample embeddings")

if not result or not result.data_array:
    print("❌ Could not fetch embeddings for diversity check")
    exit(1)

# Convert to numpy arrays (SDK returns as JSON strings)
embeddings = []
for row in result.data_array:
    emb = row[0]
    if emb is not None:
        # SDK returns embeddings as JSON strings
        if isinstance(emb, str):
            emb_list = json.loads(emb)
            embeddings.append(np.array(emb_list, dtype=np.float32))
        elif isinstance(emb, (list, tuple)):
            embeddings.append(np.array(emb, dtype=np.float32))
        else:
            embeddings.append(np.array(emb, dtype=np.float32))

if len(embeddings) == 0:
    print("❌ No embeddings found in sample")
    exit(1)

first_emb = embeddings[0]

print(f"\nFirst embedding analysis:")
print(f"  Type: {type(first_emb)}")
print(f"  Shape: {first_emb.shape}")
print(f"  Length: {len(first_emb)}")
print(f"  Sum: {np.sum(first_emb):.6f}")
print(f"  L2 norm: {np.linalg.norm(first_emb):.6f}")
print(f"  Min: {np.min(first_emb):.6f}, Max: {np.max(first_emb):.6f}")

# Check if all identical
all_same = all(np.allclose(emb, first_emb, rtol=1e-9) for emb in embeddings)

if all_same:
    print("\n❌ CRITICAL PROBLEM: All embeddings are IDENTICAL!")
    print("   df2_items has the same issue as deepfashion2_clip_embeddings")
    print("   Cannot proceed - need to regenerate embeddings using CLIP model")
    exit(1)

print("\n✅ EXCELLENT: Embeddings are UNIQUE and DIVERSE!")

# Calculate diversity metrics
similarities = [np.dot(first_emb, emb) for emb in embeddings[1:20]]
print(f"\nDiversity metrics (19 pairwise comparisons):")
print(f"  Mean similarity:   {np.mean(similarities):.4f}")
print(f"  Median similarity: {np.median(similarities):.4f}")
print(f"  Std deviation:     {np.std(similarities):.4f}")
print(f"  Range:             [{np.min(similarities):.4f}, {np.max(similarities):.4f}]")

if np.std(similarities) > 0.05:
    print(f"\n🎉 Great diversity! These embeddings are good to use.")
else:
    print(f"\n⚠️  Low diversity - embeddings may be too similar")

# Step 3: Sample image paths to understand pattern
print("\n" + "="*80)
print("STEP 3: ANALYZING IMAGE PATH PATTERNS")
print("="*80)

result = execute_sql("""
    SELECT image_path
    FROM main.fashion_demo.df2_items
    LIMIT 5
""", "Fetching sample image paths")

if result and result.data_array:
    print("\nSample image paths:")
    for row in result.data_array:
        print(f"  {row[0]}")

# Step 4: Rebuild df2_working_set with extracted item_uid
print("\n" + "="*80)
print("STEP 4: REBUILDING df2_working_set")
print("="*80)

execute_sql("""
    CREATE OR REPLACE TABLE main.fashion_demo.df2_working_set AS
    SELECT
        regexp_extract(image_path, '([0-9]+)\\.jpg$', 1) as item_uid,
        regexp_extract(image_path, '([0-9]+\\.jpg)$', 1) as filename,
        image_path,
        clip_embedding,
        current_timestamp() as created_at
    FROM main.fashion_demo.df2_items
    WHERE clip_embedding IS NOT NULL
      AND SIZE(clip_embedding) = 512
      AND regexp_extract(image_path, '([0-9]+)\\.jpg$', 1) != ''
""", "Creating new df2_working_set table")

# Step 5: Validate new working set
print("\n" + "="*80)
print("STEP 5: VALIDATING NEW WORKING SET")
print("="*80)

result = execute_sql("""
    SELECT
        COUNT(*) as total_rows,
        COUNT(DISTINCT item_uid) as unique_items,
        COUNT(clip_embedding) as has_embedding,
        COUNT(CASE WHEN SIZE(clip_embedding) = 512 THEN 1 END) as valid_512d,
        COUNT(image_path) as has_image_path,
        COUNT(filename) as has_filename
    FROM main.fashion_demo.df2_working_set
""", "Checking new df2_working_set")

if result and result.data_array:
    stats = result.data_array[0]
    new_count = int(stats[0])
    print(f"\nNew df2_working_set Statistics:")
    print(f"  Total rows:        {int(stats[0]):>10,}")
    print(f"  Unique items:      {int(stats[1]):>10,}")
    print(f"  Has embeddings:    {int(stats[2]):>10,}")
    print(f"  Valid 512D:        {int(stats[3]):>10,}")
    print(f"  Has image paths:   {int(stats[4]):>10,}")
    print(f"  Has filenames:     {int(stats[5]):>10,}")

# Show sample
result = execute_sql("""
    SELECT item_uid, filename, SIZE(clip_embedding) as emb_size
    FROM main.fashion_demo.df2_working_set
    LIMIT 5
""", "Fetching sample records")

if result and result.data_array:
    print("\nSample records:")
    print(f"  {'item_uid':<15} {'filename':<15} {'emb_size':>10}")
    print(f"  {'-'*15} {'-'*15} {'-'*10}")
    for row in result.data_array:
        print(f"  {str(row[0]):<15} {str(row[1]):<15} {int(row[2]):>10}")

# Step 6: Final embedding diversity check on new working set
print("\n" + "="*80)
print("STEP 6: FINAL DIVERSITY CHECK ON NEW WORKING SET")
print("="*80)

result = execute_sql("""
    SELECT item_uid, clip_embedding
    FROM main.fashion_demo.df2_working_set
    LIMIT 50
""", "Fetching embeddings from new working set")

if result and result.data_array:
    # Parse embeddings (SDK returns as JSON strings)
    final_embs = []
    for row in result.data_array:
        emb = row[1]
        if isinstance(emb, str):
            final_embs.append(np.array(json.loads(emb), dtype=np.float32))
        elif isinstance(emb, (list, tuple)):
            final_embs.append(np.array(emb, dtype=np.float32))
        else:
            final_embs.append(np.array(emb, dtype=np.float32))

    first_final = final_embs[0]
    all_same_final = all(np.allclose(emb, first_final, rtol=1e-9) for emb in final_embs)

    if all_same_final:
        print("❌ PROBLEM: Embeddings are still identical!")
        print("   Something went wrong.")
        exit(1)
    else:
        print("✅ SUCCESS: Embeddings are diverse!")

        final_sims = [np.dot(first_final, emb) for emb in final_embs[1:20]]
        print(f"\nFinal diversity metrics:")
        print(f"  Mean similarity:   {np.mean(final_sims):.4f}")
        print(f"  Std deviation:     {np.std(final_sims):.4f}")
        print(f"  Range:             [{np.min(final_sims):.4f}, {np.max(final_sims):.4f}]")

# Step 7: Test with Vector Search
print("\n" + "="*80)
print("STEP 7: TESTING WITH VECTOR SEARCH")
print("="*80)

try:
    vsc = VectorSearchClient(disable_notice=True)
    vs_index = vsc.get_index(
        endpoint_name="one-env-shared-endpoint-11",
        index_name="main.fashion_demo.vs_product_hybrid_search"
    )
    print("✅ Connected to Vector Search\n")
except Exception as e:
    print(f"❌ Vector Search connection failed: {e}")
    exit(1)

# Test with 3 different items
result = execute_sql("""
    SELECT item_uid, clip_embedding
    FROM main.fashion_demo.df2_working_set
    LIMIT 3
""", "Fetching 3 test items")

if not result or not result.data_array:
    print("❌ Could not fetch test items")
    exit(1)

print("Testing 3 different DF2 items...\n")
all_results = []

for idx, row in enumerate(result.data_array, 1):
    item_uid = row[0]
    embedding_raw = row[1]

    # Parse embedding (SDK returns as JSON string)
    if isinstance(embedding_raw, str):
        embedding = json.loads(embedding_raw)
    elif isinstance(embedding_raw, (list, tuple)):
        embedding = list(embedding_raw)
    else:
        embedding = list(embedding_raw)

    print(f"{idx}. Testing item_uid: {item_uid}")

    try:
        results = vs_index.similarity_search(
            query_vector=embedding,
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
    except Exception as e:
        print(f"   Error: {e}")

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

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
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
  1. Re-run Phase 2 test: python3 run_phase2_test.py
  2. Verify different top-5 results for each DF2 item
  3. If good, run full Phase 2 for all {new_count:,} items
""")
