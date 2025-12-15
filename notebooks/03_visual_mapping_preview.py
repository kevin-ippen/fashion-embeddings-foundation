# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2 Visual Preview: DF2 Items → Product Mappings
# MAGIC
# MAGIC Interactive visual preview of 20 random DeepFashion2 items with their top-5 mapped products from the catalog.
# MAGIC
# MAGIC **Fresh Data:**
# MAGIC - 1,000 DF2 images with multimodal CLIP embeddings (image + category text)
# MAGIC - 5,000 mappings to product catalog via Vector Search
# MAGIC - Categories: Tops, Dresses, Trousers, Shirts, etc.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Number of DF2 items to visualize
NUM_ITEMS = 20

# Databricks workspace host for image URLs
WORKSPACE_HOST = "https://adb-984752964297111.11.azuredatabricks.net"

# Volume path for DF2 images
DF2_VOLUME_PATH = "/Volumes/main/fashion_demo/deepfashion2_fresh/images"

# Tables
DF2_TABLE = "main.fashion_demo.df2_working_set"
MAPPING_TABLE = "main.fashion_demo.df2_to_product_mappings"
PRODUCT_TABLE = "main.fashion_demo.products_working_set"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Sample Data

# COMMAND ----------

# Get 20 random DF2 items that have mappings
sample_df2 = spark.sql(f"""
    SELECT DISTINCT df2_item_uid
    FROM {MAPPING_TABLE}
    ORDER BY RAND()
    LIMIT {NUM_ITEMS}
""")

uids = [row.df2_item_uid for row in sample_df2.collect()]
print(f"Selected {len(uids)} DF2 items for visualization:")
for i, uid in enumerate(uids, 1):
    print(f"  {i}. {uid}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Mappings and Product Info

# COMMAND ----------

# Query top-5 mapped products for each DF2 item, including image paths
uid_list = "','".join(uids)
mappings_query = f"""
    SELECT
        m.df2_item_uid,
        m.product_id,
        m.product_display_name,
        m.article_type,
        m.master_category,
        m.similarity_score,
        m.rank,
        p.image_path as product_image_path
    FROM {MAPPING_TABLE} m
    LEFT JOIN {PRODUCT_TABLE} p ON m.product_id = p.product_id
    WHERE m.df2_item_uid IN ('{uid_list}')
      AND m.rank <= 5
    ORDER BY m.df2_item_uid, m.rank
"""

mappings_df = spark.sql(mappings_query)
top5_pd = mappings_df.toPandas()

print(f"Loaded {len(top5_pd)} product mappings")
print(f"Sample:\n{top5_pd.head()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load DF2 Image Paths

# COMMAND ----------

# Get DF2 image paths
df2_query = f"""
    SELECT item_uid, filename, image_path
    FROM {DF2_TABLE}
    WHERE item_uid IN ('{uid_list}')
"""

df2_info = spark.sql(df2_query).toPandas()
print(f"Loaded {len(df2_info)} DF2 image paths")
print(f"Sample paths:")
for _, row in df2_info.head(3).iterrows():
    print(f"  {row['item_uid']}: {row['filename']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Build Image URLs and Load Images

# COMMAND ----------

import base64
from pyspark.sql.functions import col

def load_image_as_base64(image_path):
    """Load image from Volume and convert to base64 for inline display"""
    try:
        # Remove dbfs: prefix if present
        clean_path = image_path.replace('dbfs:', '')

        # Read image using Spark
        binary_df = spark.read.format("binaryFile").load(clean_path)
        binary_data = binary_df.select("content").first()

        if binary_data:
            image_bytes = bytes(binary_data[0])
            # Convert to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/jpeg;base64,{image_b64}"
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
    return None

def to_dbfs_url(image_path):
    """Convert image_path to Databricks file API URL"""
    if image_path and isinstance(image_path, str):
        # Remove 'dbfs:' prefix if present
        clean_path = image_path.replace('dbfs:', '')
        # Build file API URL
        return f"{WORKSPACE_HOST}/ajax-api/2.0/fs/files{clean_path}"
    return None

def product_img_url(path):
    """Convert product path to Databricks file API URL"""
    if path and not str(path).startswith("http"):
        if str(path).startswith("/Volumes"):
            return f"{WORKSPACE_HOST}/ajax-api/2.0/fs/files{path}"
        return path
    return path

# Load DF2 images as base64 (avoids file API permission issues)
print("Loading DF2 images from Volume...")
df2_info['df2_image_data'] = df2_info['image_path'].apply(load_image_as_base64)
df2_url_map = dict(zip(df2_info['item_uid'], df2_info['df2_image_data']))

# Convert product image paths to URLs (these work with file API)
top5_pd['product_image_url'] = top5_pd['product_image_path'].apply(product_img_url)

print("✅ Images loaded")
loaded_count = sum(1 for v in df2_url_map.values() if v is not None)
print(f"   DF2 images: {loaded_count}/{len(df2_url_map)} loaded as base64")
print(f"   Product images: Using file API URLs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate HTML Visualization

# COMMAND ----------

# Build HTML preview
html = '''
<style>
    .df2-container {
        margin-bottom: 40px;
        padding: 20px;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        background: #fafafa;
    }
    .df2-header {
        font-size: 18px;
        font-weight: bold;
        color: #0073e6;
        margin-bottom: 16px;
    }
    .df2-source-img {
        height: 150px;
        border: 3px solid #0073e6;
        margin-right: 20px;
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .products-header {
        font-weight: bold;
        margin: 16px 0 8px 0;
        color: #333;
    }
    .product-img {
        height: 120px;
        margin: 4px;
        border: 1px solid #ccc;
        border-radius: 4px;
        transition: transform 0.2s;
    }
    .product-img:hover {
        transform: scale(1.05);
        border-color: #0073e6;
    }
    .no-img {
        display: inline-block;
        width: 120px;
        height: 120px;
        background: #eee;
        color: #888;
        text-align: center;
        line-height: 120px;
        vertical-align: top;
        margin: 4px;
        border-radius: 4px;
    }
    .product-info {
        font-size: 11px;
        display: block;
        max-width: 120px;
        margin: 0 4px;
        color: #666;
    }
</style>
<div style="font-family:Arial, sans-serif; max-width:1200px;">
    <h2 style="color:#0073e6;">Phase 2 Mapping Visualization</h2>
    <p style="color:#666;">DeepFashion2 items (left) mapped to top-5 similar products (right) via Vector Search</p>
'''

for idx, uid in enumerate(uids, 1):
    html += f'<div class="df2-container">'
    html += f'<div class="df2-header">{idx}. DF2 Item: {uid}</div>'

    # DF2 source image
    df2_img = df2_url_map.get(uid)
    if df2_img:
        html += f'<img src="{df2_img}" class="df2-source-img" title="DF2 Item {uid}">'
    else:
        html += '<div class="df2-source-img" style="background:#eee;display:inline-block;">No Image</div>'

    # Top 5 products
    html += '<div style="display:inline-block; vertical-align:top;">'
    html += '<div class="products-header">→ Top 5 Similar Products:</div>'

    subset = top5_pd[top5_pd['df2_item_uid'] == uid].sort_values('rank')
    for _, row in subset.iterrows():
        img_url = row['product_image_url']
        rank = int(row['rank'])
        name = str(row['product_display_name'])[:30]
        article = str(row['article_type'])
        category = str(row['master_category'])
        score = float(row['similarity_score'])

        title_text = f"#{rank} | {name} | {article} ({category}) | sim: {score:.3f}"

        if img_url and str(img_url) != 'nan':
            html += f'<div style="display:inline-block; text-align:center; vertical-align:top;">'
            html += f'<img src="{img_url}" class="product-img" title="{title_text}">'
            html += f'<span class="product-info">#{rank} · {score:.3f}</span>'
            html += f'</div>'
        else:
            html += f'<div style="display:inline-block; text-align:center;">'
            html += f'<div class="no-img" title="{title_text}">No Image</div>'
            html += f'<span class="product-info">#{rank} · {score:.3f}</span>'
            html += f'</div>'

    html += '</div>'  # Close products container
    html += '</div>'  # Close df2-container

html += '</div>'

displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Detailed Mapping Table

# COMMAND ----------

# Display detailed table
display(top5_pd[[
    'df2_item_uid', 'rank', 'product_id', 'product_display_name',
    'article_type', 'master_category', 'similarity_score'
]].sort_values(['df2_item_uid', 'rank']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Similarity Score Analysis

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of similarity scores
axes[0].hist(top5_pd['similarity_score'], bins=20, edgecolor='black', color='steelblue')
axes[0].set_xlabel('Similarity Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Distribution of Similarity Scores ({len(top5_pd)} mappings)')
axes[0].axvline(top5_pd['similarity_score'].mean(), color='red', linestyle='--',
                label=f'Mean: {top5_pd["similarity_score"].mean():.3f}')
axes[0].legend()

# Similarity by rank
rank_scores = top5_pd.groupby('rank')['similarity_score'].agg(['mean', 'std']).reset_index()
axes[1].bar(rank_scores['rank'], rank_scores['mean'], yerr=rank_scores['std'],
            capsize=5, color='steelblue', edgecolor='black')
axes[1].set_xlabel('Rank')
axes[1].set_ylabel('Avg Similarity Score')
axes[1].set_title('Average Similarity by Rank')
axes[1].set_xticks([1, 2, 3, 4, 5])

plt.tight_layout()
plt.show()

print(f"\n📊 Score Statistics:")
print(f"  Mean:   {top5_pd['similarity_score'].mean():.4f}")
print(f"  Median: {top5_pd['similarity_score'].median():.4f}")
print(f"  Std:    {top5_pd['similarity_score'].std():.4f}")
print(f"  Range:  [{top5_pd['similarity_score'].min():.4f}, {top5_pd['similarity_score'].max():.4f}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Category Analysis

# COMMAND ----------

# Count article types in mapped products
article_counts = top5_pd['article_type'].value_counts().head(10)

plt.figure(figsize=(12, 6))
article_counts.plot(kind='barh', color='steelblue', edgecolor='black')
plt.xlabel('Count')
plt.title(f'Top 10 Article Types in Mapped Products (n={len(top5_pd)})')
plt.tight_layout()
plt.show()

print(f"\n📦 Article Type Distribution:")
for article, count in article_counts.items():
    pct = 100 * count / len(top5_pd)
    print(f"  {article:30s} {count:>4} ({pct:>5.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook visualizes the Phase 2 Vector Search mapping results using **fresh clean data**:
# MAGIC
# MAGIC **Dataset:**
# MAGIC - 1,000 DeepFashion2 images with multimodal CLIP embeddings (image + category text)
# MAGIC - 5,000 total mappings (1,000 items × top-5 products each)
# MAGIC - Fresh data from Kaggle with verified embeddings
# MAGIC
# MAGIC **Visualization:**
# MAGIC - Shows 20 random DF2 items with their top-5 mapped products
# MAGIC - Displays similarity scores for each mapping
# MAGIC - Analyzes score distribution and article type coverage
# MAGIC
# MAGIC **Quality Metrics:**
# MAGIC - Avg similarity: ~0.56 (reasonable matches)
# MAGIC - Different items return different products (no duplicate issue)
# MAGIC - Category-aware matching (tops→tops, dresses→dresses, etc.)
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Review visual quality of mappings
# MAGIC - Validate semantic correctness (do the matches make sense?)
# MAGIC - Consider re-encoding products with multimodal for better diversity
# MAGIC - Proceed to Phase 3: Graph Construction using these mappings
