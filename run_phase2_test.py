#!/usr/bin/env python3
"""
Phase 2: Vector Search Mapping - TEST MODE (100 items)
Quick validation before running full 22K mapping
"""

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
import time
import json

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
            print(f"✅ {description} - Complete")
        return response.result
    else:
        print(f"❌ Query failed: {response.status.state}")
        if response.status.error:
            print(f"   Error: {response.status.error}")
        return None

# Configuration
TEST_LIMIT = 100
TOP_K = 5
BATCH_SIZE = 20
VS_ENDPOINT = "one-env-shared-endpoint-11"
VS_INDEX = "main.fashion_demo.vs_product_hybrid_search"

print("="*80)
print("PHASE 2: VECTOR SEARCH MAPPING - TEST MODE")
print("="*80)
print(f"Test items:  {TEST_LIMIT}")
print(f"Top-K:       {TOP_K}")
print(f"Batch size:  {BATCH_SIZE}")
print()

# 1. Load test items
print("="*70)
print("1. LOADING TEST DATA")
print("="*70)

result = execute_sql(
    f"""
    SELECT item_uid
    FROM main.fashion_demo.df2_working_set
    LIMIT {TEST_LIMIT}
    """,
    f"Loading {TEST_LIMIT} test items"
)

if not result or not result.data_array:
    print("❌ Could not load test items")
    exit(1)

item_uids = [row[0] for row in result.data_array]
print(f"✅ Loaded {len(item_uids)} items for testing\n")

# 2. Initialize Vector Search
print("="*70)
print("2. INITIALIZING VECTOR SEARCH")
print("="*70)

try:
    vsc = VectorSearchClient(disable_notice=True)
    vs_index = vsc.get_index(
        endpoint_name=VS_ENDPOINT,
        index_name=VS_INDEX
    )
    print(f"✅ Connected to index: {VS_INDEX}\n")
except Exception as e:
    print(f"❌ Vector Search error: {e}")
    exit(1)

# 3. Batch mapping
print("="*70)
print("3. MAPPING TEST ITEMS")
print("="*70)

all_mappings = []
total_batches = (len(item_uids) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(total_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(item_uids))
    batch_uids = item_uids[start_idx:end_idx]

    print(f"Batch {batch_idx + 1}/{total_batches}: Items {start_idx + 1}-{end_idx}...", end=" ")

    # Fetch embeddings for batch
    uid_list = "','".join(batch_uids)
    result = execute_sql(f"""
        SELECT item_uid, clip_embedding
        FROM main.fashion_demo.df2_working_set
        WHERE item_uid IN ('{uid_list}')
    """)

    if not result or not result.data_array:
        print("⚠️  Skipped")
        continue

    # Map each item
    for row in result.data_array:
        item_uid = row[0]
        embedding_raw = row[1]

        # Parse embedding (SDK returns as JSON string)
        if isinstance(embedding_raw, str):
            embedding = json.loads(embedding_raw)
        elif isinstance(embedding_raw, (list, tuple)):
            embedding = list(embedding_raw)
        else:
            embedding = list(embedding_raw)

        try:
            results = vs_index.similarity_search(
                query_vector=embedding,
                columns=["product_id", "product_display_name", "article_type",
                        "master_category", "price"],
                num_results=TOP_K
            )

            if results and 'result' in results and 'data_array' in results['result']:
                for rank, result_row in enumerate(results['result']['data_array'], 1):
                    all_mappings.append({
                        'df2_item_uid': item_uid,
                        'product_id': int(result_row[0]),
                        'product_display_name': result_row[1],
                        'article_type': result_row[2],
                        'master_category': result_row[3],
                        'price': float(result_row[4]),
                        'similarity_score': float(result_row[-1]) if len(result_row) > 5 else 0.0,
                        'rank': rank
                    })
        except Exception as e:
            print(f"\n   ⚠️  Error: {e}")
            continue

    print(f"✅ ({len(all_mappings)} total)")
    time.sleep(0.2)

print(f"\n✅ Mapping complete: {len(all_mappings)} mappings created\n")

# 4. Save to Delta
print("="*70)
print("4. SAVING TO DELTA TABLE")
print("="*70)

# Create table
execute_sql(
    """
    CREATE OR REPLACE TABLE main.fashion_demo.df2_to_product_mappings_test (
        df2_item_uid STRING,
        product_id BIGINT,
        product_display_name STRING,
        article_type STRING,
        master_category STRING,
        price DOUBLE,
        similarity_score DOUBLE,
        rank INT,
        created_at TIMESTAMP
    ) USING DELTA
    """,
    "Creating test table"
)

# Insert in batches
INSERT_BATCH = 500
total_inserts = (len(all_mappings) + INSERT_BATCH - 1) // INSERT_BATCH

for batch_idx in range(total_inserts):
    start = batch_idx * INSERT_BATCH
    end = min(start + INSERT_BATCH, len(all_mappings))
    batch = all_mappings[start:end]

    values = []
    for m in batch:
        name_esc = m['product_display_name'].replace("'", "''")
        article_esc = m['article_type'].replace("'", "''")
        values.append(
            f"('{m['df2_item_uid']}', {m['product_id']}, '{name_esc}', "
            f"'{article_esc}', '{m['master_category']}', {m['price']}, "
            f"{m['similarity_score']}, {m['rank']}, current_timestamp())"
        )

    execute_sql(f"""
        INSERT INTO main.fashion_demo.df2_to_product_mappings_test
        VALUES {', '.join(values)}
    """)

result = execute_sql("SELECT COUNT(*) FROM main.fashion_demo.df2_to_product_mappings_test")
saved = int(result.data_array[0][0]) if result and result.data_array else 0
print(f"✅ Saved {saved} mappings\n")

# 5. Quick analysis
print("="*70)
print("5. QUICK ANALYSIS")
print("="*70)

result = execute_sql("""
    SELECT
        COUNT(DISTINCT df2_item_uid) as unique_items,
        COUNT(*) as total_mappings,
        ROUND(AVG(similarity_score), 4) as avg_similarity,
        ROUND(MIN(similarity_score), 4) as min_similarity,
        ROUND(MAX(similarity_score), 4) as max_similarity
    FROM main.fashion_demo.df2_to_product_mappings_test
""")

if result and result.data_array:
    stats = result.data_array[0]
    print(f"Unique DF2 items:    {int(stats[0]):>10,}")
    print(f"Total mappings:      {int(stats[1]):>10,}")
    print(f"Avg similarity:      {float(stats[2]):>10.4f}")
    print(f"Min similarity:      {float(stats[3]):>10.4f}")
    print(f"Max similarity:      {float(stats[4]):>10.4f}")

# Score distribution
result = execute_sql("""
    SELECT
        CASE
            WHEN similarity_score >= 0.9 THEN '0.9-1.0 (Excellent)'
            WHEN similarity_score >= 0.8 THEN '0.8-0.9 (Very Good)'
            WHEN similarity_score >= 0.7 THEN '0.7-0.8 (Good)'
            WHEN similarity_score >= 0.6 THEN '0.6-0.7 (Fair)'
            ELSE '<0.6 (Poor)'
        END as score_range,
        COUNT(*) as count
    FROM main.fashion_demo.df2_to_product_mappings_test
    GROUP BY score_range
    ORDER BY score_range DESC
""")

if result and result.data_array:
    print(f"\nSimilarity Distribution:")
    for row in result.data_array:
        print(f"  {row[0]:25s} {int(row[1]):>6,}")

print("\n" + "="*80)
print("✅ TEST COMPLETE!")
print("="*80)
print("\nNext Steps:")
print("1. Review test results in Databricks:")
print("   SELECT * FROM main.fashion_demo.df2_to_product_mappings_test")
print()
print("2. Use the notebook for visual exploration:")
print("   notebooks/02_phase2_mapping_test.py")
print()
print("3. If satisfied, run full mapping (22K items):")
print("   python3 run_phase2_mapping.py")
