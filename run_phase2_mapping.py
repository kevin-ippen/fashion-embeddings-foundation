#!/usr/bin/env python3
"""
Phase 2: Vector Search Mapping - Databricks SDK Execution Script
Maps DeepFashion2 items to product catalog using Vector Search
"""

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
import time
import json

# Initialize Databricks client
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
    else:
        raise Exception("No SQL warehouse found")

warehouse_id = warehouse.id
print(f"✅ Using warehouse: {warehouse.name} ({warehouse_id})\n")

def to_int(val):
    """Convert value to int, handling strings"""
    if isinstance(val, str):
        return int(val)
    return int(val) if val is not None else 0

def to_float(val):
    """Convert value to float, handling strings"""
    if isinstance(val, str):
        return float(val)
    return float(val) if val is not None else 0.0

def execute_sql(query, description=None):
    """Execute SQL query and return results"""
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

# ============================================================================
# PHASE 2: VECTOR SEARCH MAPPING
# ============================================================================

print("="*80)
print("PHASE 2: VECTOR SEARCH MAPPING")
print("="*80)
print()

# ============================================================================
# 1. LOAD DEEPFASHION2 EMBEDDINGS
# ============================================================================

print("="*70)
print("1. LOADING DEEPFASHION2 WORKING SET")
print("="*70)

# Get count
result = execute_sql(
    "SELECT COUNT(*) FROM main.fashion_demo.df2_working_set",
    "Counting DeepFashion2 items"
)
df2_count = to_int(result.data_array[0][0]) if result and result.data_array else 0
print(f"Total DeepFashion2 items to map: {df2_count:,}\n")

# ============================================================================
# 2. INITIALIZE VECTOR SEARCH
# ============================================================================

print("="*70)
print("2. INITIALIZING VECTOR SEARCH")
print("="*70)

try:
    vsc = VectorSearchClient(disable_notice=True)
    print("✅ Vector Search client initialized")

    # Try to get the index
    vs_index = vsc.get_index(
        endpoint_name="one-env-shared-endpoint-11",
        index_name="main.fashion_demo.vs_product_hybrid_search"
    )
    print(f"✅ Connected to index: main.fashion_demo.vs_product_hybrid_search\n")

    vector_search_available = True
except Exception as e:
    print(f"❌ Vector Search not available: {e}")
    print("   Cannot proceed with Phase 2 without Vector Search")
    vector_search_available = False

if not vector_search_available:
    print("\n⚠️  Phase 2 requires Vector Search to be available.")
    print("   Please ensure the Vector Search endpoint is running.")
    exit(1)

# ============================================================================
# 3. BATCH MAPPING VIA VECTOR SEARCH
# ============================================================================

print("="*70)
print("3. MAPPING DF2 ITEMS TO PRODUCTS")
print("="*70)

# Fetch item UIDs first (lightweight)
print("Fetching DeepFashion2 item UIDs...")
result = execute_sql(
    """
    SELECT item_uid
    FROM main.fashion_demo.df2_working_set
    ORDER BY item_uid
    """,
    "Loading DeepFashion2 item UIDs"
)

if not result or not result.data_array:
    print("❌ Could not load DeepFashion2 item UIDs")
    exit(1)

item_uids = [row[0] for row in result.data_array]
print(f"✅ Loaded {len(item_uids):,} item UIDs\n")

# Mapping configuration
TOP_K = 5  # Top-5 products per DF2 item
FETCH_BATCH_SIZE = 50  # Fetch embeddings in batches of 50
SLEEP_BETWEEN_BATCHES = 0.5  # Rate limiting

all_mappings = []
total_batches = (len(item_uids) + FETCH_BATCH_SIZE - 1) // FETCH_BATCH_SIZE

print(f"Starting batch mapping:")
print(f"  Items to process: {len(item_uids):,}")
print(f"  Top-K per item: {TOP_K}")
print(f"  Fetch batch size: {FETCH_BATCH_SIZE}")
print(f"  Total batches: {total_batches}")
print()

# Process in batches
for batch_idx in range(total_batches):
    start_idx = batch_idx * FETCH_BATCH_SIZE
    end_idx = min(start_idx + FETCH_BATCH_SIZE, len(item_uids))
    batch_uids = item_uids[start_idx:end_idx]

    print(f"⏳ Batch {batch_idx + 1}/{total_batches}: Processing items {start_idx + 1}-{end_idx}...")

    # Fetch embeddings for this batch
    uid_list = "','".join(batch_uids)
    fetch_query = f"""
        SELECT item_uid, clip_embedding
        FROM main.fashion_demo.df2_working_set
        WHERE item_uid IN ('{uid_list}')
    """

    result = execute_sql(fetch_query)
    if not result or not result.data_array:
        print(f"   ⚠️  Could not fetch batch {batch_idx + 1}, skipping...")
        continue

    # Map each item in the batch
    for row in result.data_array:
        item_uid = row[0]
        embedding = row[1]

        try:
            # Query Vector Search
            results = vs_index.similarity_search(
                query_vector=embedding,
                columns=["product_id", "product_display_name", "article_type", "master_category", "price"],
                num_results=TOP_K
            )

            # Extract results
            if results and 'result' in results and 'data_array' in results['result']:
                for rank, result_row in enumerate(results['result']['data_array'], 1):
                    product_id = to_int(result_row[0])
                    product_name = result_row[1]
                    article_type = result_row[2]
                    master_category = result_row[3]
                    price = to_float(result_row[4])
                    similarity_score = to_float(result_row[-1]) if len(result_row) > 5 else 0.0

                    all_mappings.append({
                        'df2_item_uid': item_uid,
                        'product_id': product_id,
                        'product_display_name': product_name,
                        'article_type': article_type,
                        'master_category': master_category,
                        'price': price,
                        'similarity_score': similarity_score,
                        'rank': rank
                    })
        except Exception as e:
            print(f"   ⚠️  Error mapping item {item_uid}: {e}")
            continue

    print(f"✅ Batch {batch_idx + 1}/{total_batches} complete ({len(all_mappings):,} total mappings)")

    # Rate limiting
    if batch_idx < total_batches - 1:
        time.sleep(SLEEP_BETWEEN_BATCHES)

print(f"\n✅ Mapping complete!")
print(f"   Total mappings created: {len(all_mappings):,}")
print(f"   Expected mappings: {df2_count * TOP_K:,}")
print(f"   Coverage: {(len(all_mappings) / (df2_count * TOP_K) * 100):.1f}%\n")

# ============================================================================
# 4. SAVE MAPPINGS TO DELTA TABLE
# ============================================================================

print("="*70)
print("4. SAVING MAPPINGS TO DELTA TABLE")
print("="*70)

# Create temporary table with mappings
print("Creating mappings table...")

# First, create the table structure
create_table_sql = """
CREATE OR REPLACE TABLE main.fashion_demo.df2_to_product_mappings (
    df2_item_uid STRING,
    product_id BIGINT,
    product_display_name STRING,
    article_type STRING,
    master_category STRING,
    price DOUBLE,
    similarity_score DOUBLE,
    rank INT,
    created_at TIMESTAMP
)
USING DELTA
"""

execute_sql(create_table_sql, "Creating table structure")

# Insert mappings in batches
INSERT_BATCH_SIZE = 1000
total_insert_batches = (len(all_mappings) + INSERT_BATCH_SIZE - 1) // INSERT_BATCH_SIZE

print(f"\nInserting {len(all_mappings):,} mappings in {total_insert_batches} batches...")

for batch_idx in range(total_insert_batches):
    start_idx = batch_idx * INSERT_BATCH_SIZE
    end_idx = min(start_idx + INSERT_BATCH_SIZE, len(all_mappings))
    batch_mappings = all_mappings[start_idx:end_idx]

    # Build INSERT VALUES statement
    values = []
    for m in batch_mappings:
        # Escape single quotes
        name_escaped = m['product_display_name'].replace("'", "''")
        article_escaped = m['article_type'].replace("'", "''")

        values.append(
            f"('{m['df2_item_uid']}', {m['product_id']}, "
            f"'{name_escaped}', '{article_escaped}', "
            f"'{m['master_category']}', {m['price']}, "
            f"{m['similarity_score']}, {m['rank']}, current_timestamp())"
        )

    insert_sql = f"""
        INSERT INTO main.fashion_demo.df2_to_product_mappings
        VALUES {', '.join(values)}
    """

    result = execute_sql(insert_sql, f"Inserting batch {batch_idx + 1}/{total_insert_batches}")

# Verify count
result = execute_sql("SELECT COUNT(*) FROM main.fashion_demo.df2_to_product_mappings")
saved_count = to_int(result.data_array[0][0]) if result and result.data_array else 0
print(f"\n✅ Saved {saved_count:,} mappings to Delta table\n")

# ============================================================================
# 5. QUALITY ANALYSIS
# ============================================================================

print("="*70)
print("5. QUALITY ANALYSIS")
print("="*70)

# Mapping coverage
result = execute_sql(
    """
    SELECT
        COUNT(DISTINCT df2_item_uid) as unique_df2_items,
        COUNT(*) as total_mappings,
        AVG(similarity_score) as avg_similarity,
        MIN(similarity_score) as min_similarity,
        MAX(similarity_score) as max_similarity
    FROM main.fashion_demo.df2_to_product_mappings
    """,
    "Computing mapping statistics"
)

if result and result.data_array:
    stats = result.data_array[0]
    unique_df2 = to_int(stats[0])
    total_maps = to_int(stats[1])
    avg_sim = to_float(stats[2])
    min_sim = to_float(stats[3])
    max_sim = to_float(stats[4])

    print(f"\nMapping Statistics:")
    print(f"  Unique DF2 items mapped:  {unique_df2:>10,}")
    print(f"  Total mappings:           {total_maps:>10,}")
    print(f"  Avg similarity score:     {avg_sim:>10.4f}")
    print(f"  Min similarity score:     {min_sim:>10.4f}")
    print(f"  Max similarity score:     {max_sim:>10.4f}")
    print(f"  Coverage:                 {(unique_df2 / df2_count * 100):>10.1f}%")

# Similarity score distribution
result = execute_sql(
    """
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
    FROM main.fashion_demo.df2_to_product_mappings
    GROUP BY score_range
    ORDER BY score_range DESC
    """,
    "Analyzing similarity score distribution"
)

if result and result.data_array:
    print(f"\n Similarity Score Distribution:")
    for row in result.data_array:
        range_label = row[0]
        count = to_int(row[1])
        pct = to_float(row[2])
        print(f"  {range_label:25s} {count:>8,} ({pct:>5.1f}%)")

# Top mapped products
result = execute_sql(
    """
    SELECT
        product_id,
        product_display_name,
        article_type,
        COUNT(*) as map_count,
        AVG(similarity_score) as avg_score
    FROM main.fashion_demo.df2_to_product_mappings
    GROUP BY product_id, product_display_name, article_type
    ORDER BY map_count DESC
    LIMIT 10
    """,
    "Finding most frequently mapped products"
)

if result and result.data_array:
    print(f"\nTop 10 Most Mapped Products:")
    print(f"{'Product ID':>10} {'Name':40} {'Type':20} {'Count':>8} {'Avg Score':>10}")
    print("-" * 95)
    for row in result.data_array:
        pid = to_int(row[0])
        name = row[1][:37] + '...' if len(row[1]) > 40 else row[1]
        art_type = row[2][:17] + '...' if len(row[2]) > 20 else row[2]
        map_count = to_int(row[3])
        avg_score = to_float(row[4])
        print(f"{pid:>10} {name:40} {art_type:20} {map_count:>8,} {avg_score:>10.4f}")

# ============================================================================
# 6. PHASE 2 SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PHASE 2 MAPPING - SUMMARY REPORT")
print("="*80)

print(f"\n✅ VECTOR SEARCH MAPPING")
print(f"   DeepFashion2 items:      {df2_count:>10,}")
print(f"   Items mapped:            {unique_df2:>10,}")
print(f"   Total mappings:          {total_maps:>10,} (top-{TOP_K})")
print(f"   Coverage:                {(unique_df2 / df2_count * 100):>10.1f}%")
print(f"   Avg similarity:          {avg_sim:>10.4f}")

print(f"\n✅ DELTA TABLE CREATED")
print(f"   Table: main.fashion_demo.df2_to_product_mappings")
print(f"   Rows: {saved_count:,}")

print(f"\n{'='*80}")
print(f"✅ PHASE 2 COMPLETE - Ready for Phase 3 (Graph Construction)")
print(f"{'='*80}")

# Save summary
summary_sql = f"""
    CREATE OR REPLACE TABLE main.fashion_demo.phase2_mapping_summary AS
    SELECT
        {df2_count} as df2_items_total,
        {unique_df2} as df2_items_mapped,
        {total_maps} as total_mappings,
        {TOP_K} as top_k_per_item,
        {avg_sim} as avg_similarity_score,
        current_timestamp() as mapping_timestamp,
        'PHASE_2_COMPLETE' as status
"""

execute_sql(summary_sql, "Saving mapping summary")
print("\n✅ Summary saved to main.fashion_demo.phase2_mapping_summary")

print("\n✅ Phase 2 mapping complete!")
