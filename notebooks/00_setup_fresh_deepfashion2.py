# Databricks notebook source
# MAGIC %md
# MAGIC # Setup Fresh DeepFashion2 Dataset
# MAGIC
# MAGIC Download clean DeepFashion2 data from Kaggle and prepare for CLIP encoding.
# MAGIC
# MAGIC **Dataset**: https://www.kaggle.com/datasets/thusharanair/deepfashion2-original-with-dataframes
# MAGIC
# MAGIC **Steps**:
# MAGIC 1. Install Kaggle CLI
# MAGIC 2. Download dataset
# MAGIC 3. Extract to Volume
# MAGIC 4. Create catalog tables
# MAGIC 5. Sample and validate

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Kaggle dataset
KAGGLE_DATASET = "thusharanair/deepfashion2-original-with-dataframes"

# Target Volume
TARGET_VOLUME = "/Volumes/main/fashion_demo/deepfashion2_fresh"

# Catalog tables
CATALOG = "main"
SCHEMA = "fashion_demo"
IMAGE_TABLE = f"{CATALOG}.{SCHEMA}.df2_images_fresh"
METADATA_TABLE = f"{CATALOG}.{SCHEMA}.df2_metadata_fresh"

# How many images to process initially (for testing)
INITIAL_SAMPLE = 1000

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Kaggle CLI

# COMMAND ----------

# Install kaggle package
%pip install kaggle --quiet
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Setup Kaggle Credentials

# COMMAND ----------

import os

# Get credentials from secrets
os.environ['KAGGLE_USERNAME'] = dbutils.secrets.get(scope="kagglekev", key="kevinippen")
os.environ['KAGGLE_KEY'] = dbutils.secrets.get(scope="kagglekev", key="kagglekey")

# Verify
import kaggle
print("✅ Kaggle CLI authenticated")
print(f"   Username: {os.environ['KAGGLE_USERNAME']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Download Dataset

# COMMAND ----------

import os

# Create temp download directory
download_dir = "/tmp/deepfashion2_download"
os.makedirs(download_dir, exist_ok=True)

print(f"Downloading dataset: {KAGGLE_DATASET}")
print(f"To: {download_dir}")
print("This may take 5-10 minutes...")

# Download dataset
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

api.dataset_download_files(
    KAGGLE_DATASET,
    path=download_dir,
    unzip=True
)

print(f"\n✅ Download complete!")

# List downloaded files
files = os.listdir(download_dir)
print(f"\nDownloaded {len(files)} files:")
for f in sorted(files)[:20]:
    print(f"  {f}")
if len(files) > 20:
    print(f"  ... and {len(files) - 20} more")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Explore Downloaded Data

# COMMAND ----------

import os
import glob

# Check structure
print("Dataset structure:")

# Look for image directories
image_dirs = glob.glob(f"{download_dir}/**/images", recursive=True)
if image_dirs:
    print(f"\n✅ Found image directory: {image_dirs[0]}")

    # Count images
    image_files = glob.glob(f"{image_dirs[0]}/*.jpg") + glob.glob(f"{image_dirs[0]}/*.png")
    print(f"   Total images: {len(image_files):,}")
    print(f"   Sample: {image_files[:3]}")
else:
    print("⚠️  No 'images' directory found")
    print("Available directories:")
    for root, dirs, files in os.walk(download_dir):
        level = root.replace(download_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 2:  # Only show first 2 levels
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:
                print(f"{subindent}{file}")

# Look for CSV/parquet metadata
metadata_files = (
    glob.glob(f"{download_dir}/**/*.csv", recursive=True) +
    glob.glob(f"{download_dir}/**/*.parquet", recursive=True) +
    glob.glob(f"{download_dir}/**/*.json", recursive=True)
)

if metadata_files:
    print(f"\n✅ Found {len(metadata_files)} metadata files:")
    for f in metadata_files[:10]:
        rel_path = f.replace(download_dir, '')
        print(f"   {rel_path}")
else:
    print("\n⚠️  No CSV/parquet metadata found")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Copy Images to Volume

# COMMAND ----------

import shutil
import os

# Create Volume directory if it doesn't exist
dbutils.fs.mkdirs(TARGET_VOLUME)

# Get source image directory
image_source = image_dirs[0] if image_dirs else None

if image_source:
    print(f"Copying images from: {image_source}")
    print(f"To Volume: {TARGET_VOLUME}")
    print(f"\nStarting with {INITIAL_SAMPLE} images for testing...")

    # Get image files
    all_images = sorted(glob.glob(f"{image_source}/*.jpg") + glob.glob(f"{image_source}/*.png"))

    if len(all_images) == 0:
        print("❌ No images found!")
    else:
        # Copy initial sample
        images_to_copy = all_images[:INITIAL_SAMPLE]

        # Use dbutils to copy (more reliable than shutil for Volumes)
        target_path = TARGET_VOLUME + "/images"
        dbutils.fs.mkdirs(target_path)

        copied = 0
        failed = 0

        for i, img_path in enumerate(images_to_copy):
            try:
                filename = os.path.basename(img_path)
                target = f"{target_path}/{filename}"

                # Read and write via dbutils
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()

                dbutils.fs.put(target, img_bytes.decode('latin1'), overwrite=True)
                copied += 1

                if (i + 1) % 100 == 0:
                    print(f"  Copied {i + 1}/{len(images_to_copy)}...")

            except Exception as e:
                print(f"  ⚠️  Failed to copy {filename}: {e}")
                failed += 1
                continue

        print(f"\n✅ Copied {copied} images to Volume")
        if failed > 0:
            print(f"   ⚠️  {failed} images failed")

        # Verify
        copied_files = dbutils.fs.ls(target_path)
        print(f"   Volume now has: {len(copied_files)} files")

else:
    print("❌ No image source directory found - check dataset structure")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Load Metadata

# COMMAND ----------

# Try to find and load metadata
print("Looking for metadata files...")

metadata_df = None

# Check for CSV files
csv_files = glob.glob(f"{download_dir}/**/*.csv", recursive=True)
if csv_files:
    print(f"\nFound {len(csv_files)} CSV files:")
    for csv in csv_files:
        print(f"  {os.path.basename(csv)}")

    # Try to load the first one
    try:
        import pandas as pd
        pdf = pd.read_csv(csv_files[0], nrows=5)
        print(f"\n✅ Sample from {os.path.basename(csv_files[0])}:")
        print(pdf)

        # Load full dataset
        print(f"\nLoading full CSV...")
        metadata_df = spark.read.csv(csv_files[0], header=True, inferSchema=True)
        print(f"✅ Loaded {metadata_df.count():,} rows")

    except Exception as e:
        print(f"❌ Error loading CSV: {e}")

# Check for parquet
parquet_files = glob.glob(f"{download_dir}/**/*.parquet", recursive=True)
if parquet_files and metadata_df is None:
    print(f"\nFound {len(parquet_files)} parquet files:")
    for pq in parquet_files:
        print(f"  {os.path.basename(pq)}")

    try:
        metadata_df = spark.read.parquet(parquet_files[0])
        print(f"✅ Loaded {metadata_df.count():,} rows from parquet")
    except Exception as e:
        print(f"❌ Error loading parquet: {e}")

if metadata_df:
    print(f"\nMetadata schema:")
    metadata_df.printSchema()

    print(f"\nSample data:")
    display(metadata_df.limit(10))
else:
    print("\n⚠️  No metadata loaded - will create from images")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Create Image Catalog Table

# COMMAND ----------

# Read images from Volume using Spark
images_df = spark.read.format("binaryFile").load(f"{TARGET_VOLUME}/images/*.jpg")

# Extract filename and add metadata
from pyspark.sql.functions import element_at, split, regexp_extract, current_timestamp

processed_df = images_df.select(
    element_at(split(images_df.path, "/"), -1).alias("filename"),
    regexp_extract(element_at(split(images_df.path, "/"), -1), r"(\d+)", 1).alias("image_id"),
    images_df.path.alias("image_path"),
    images_df.length.alias("file_size_bytes"),
    images_df.modificationTime.alias("file_modified_time"),
    current_timestamp().alias("imported_at")
)

# Show sample
print(f"Processed {processed_df.count():,} images")
display(processed_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save to Delta Tables

# COMMAND ----------

# Save image catalog
processed_df.write.mode("overwrite").saveAsTable(IMAGE_TABLE)

print(f"✅ Saved image catalog to: {IMAGE_TABLE}")
print(f"   Rows: {spark.table(IMAGE_TABLE).count():,}")

# If we have metadata, save it too
if metadata_df:
    metadata_df.write.mode("overwrite").saveAsTable(METADATA_TABLE)
    print(f"✅ Saved metadata to: {METADATA_TABLE}")
    print(f"   Rows: {spark.table(METADATA_TABLE).count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Validation

# COMMAND ----------

# Validate setup
print("="*70)
print("VALIDATION SUMMARY")
print("="*70)

# Check Volume
volume_files = dbutils.fs.ls(f"{TARGET_VOLUME}/images")
print(f"\n✅ Volume: {TARGET_VOLUME}/images")
print(f"   Files: {len(volume_files):,}")

# Check image table
img_count = spark.table(IMAGE_TABLE).count()
print(f"\n✅ Image Catalog: {IMAGE_TABLE}")
print(f"   Rows: {img_count:,}")

# Sample record
sample = spark.table(IMAGE_TABLE).first()
print(f"\n   Sample record:")
print(f"     Filename: {sample.filename}")
print(f"     Image ID: {sample.image_id}")
print(f"     Path: {sample.image_path}")
print(f"     Size: {sample.file_size_bytes:,} bytes")

# Check metadata if exists
try:
    meta_count = spark.table(METADATA_TABLE).count()
    print(f"\n✅ Metadata: {METADATA_TABLE}")
    print(f"   Rows: {meta_count:,}")
except:
    print(f"\n⚠️  No metadata table created")

print("\n" + "="*70)
print("SETUP COMPLETE!")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Encode with CLIP**: Run notebook to encode all images
# MAGIC 2. **Create df2_working_set**: Build clean working table with embeddings
# MAGIC 3. **Validate quality**: Check embedding diversity
# MAGIC 4. **Scale up**: Process remaining images if sample looks good
# MAGIC
# MAGIC ### To process more images:
# MAGIC ```python
# MAGIC # Modify INITIAL_SAMPLE in cell 1 and re-run cells 5-9
# MAGIC # Or set to len(all_images) to copy all
# MAGIC ```
