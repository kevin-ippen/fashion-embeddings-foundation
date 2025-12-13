"""
Vector Search Examples for Graph-Based Outfit Composition

This module demonstrates how to use Mosaic AI Vector Search for the outfit composition project.
Used in Phase 1 (validation) and Phase 2 (DeepFashion2 → Product mapping).
"""

from databricks.vector_search.client import VectorSearchClient
import requests
import base64
from typing import List, Dict, Optional


# Configuration
WORKSPACE_URL = "https://your-workspace.cloud.databricks.com"
VS_ENDPOINT = "fashion_vector_search"
CLIP_ENDPOINT = "clip-multimodal-encoder"

# Index names
IMAGE_INDEX = "main.fashion_demo.vs_image_search"
TEXT_INDEX = "main.fashion_demo.vs_text_search"
HYBRID_INDEX = "main.fashion_demo.vs_hybrid_search"


def get_text_embedding(text: str, token: str) -> List[float]:
    """
    Generate CLIP text embedding using Model Serving endpoint.

    Args:
        text: Text query (e.g., "red leather jacket")
        token: Databricks personal access token

    Returns:
        512-dimensional embedding vector
    """
    url = f"{WORKSPACE_URL}/serving-endpoints/{CLIP_ENDPOINT}/invocations"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "dataframe_records": [{"text": text}]
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    return response.json()["predictions"][0]


def get_image_embedding(image_bytes: bytes, token: str) -> List[float]:
    """
    Generate CLIP image embedding using Model Serving endpoint.

    Args:
        image_bytes: Raw image bytes
        token: Databricks personal access token

    Returns:
        512-dimensional embedding vector
    """
    url = f"{WORKSPACE_URL}/serving-endpoints/{CLIP_ENDPOINT}/invocations"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Encode image as base64
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    payload = {
        "dataframe_records": [{"image": image_b64}]
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    return response.json()["predictions"][0]


def search_by_text(
    query: str,
    token: str,
    num_results: int = 10,
    filters: Optional[Dict] = None
) -> List[Dict]:
    """
    Search products using text query.

    Args:
        query: Text query (e.g., "summer dresses")
        token: Databricks personal access token
        num_results: Number of results to return
        filters: Optional filters (e.g., {"gender": "Women", "price <": 100})

    Returns:
        List of matching products with similarity scores
    """
    # Generate text embedding
    embedding = get_text_embedding(query, token)

    # Search vector index
    vsc = VectorSearchClient()
    index = vsc.get_index(
        endpoint_name=VS_ENDPOINT,
        index_name=TEXT_INDEX
    )

    results = index.similarity_search(
        query_vector=embedding,
        columns=[
            "product_id",
            "product_display_name",
            "master_category",
            "sub_category",
            "base_color",
            "price",
            "image_path"
        ],
        num_results=num_results,
        filters=filters
    )

    return results["result"]["data_array"]


def search_by_image(
    image_bytes: bytes,
    token: str,
    num_results: int = 10,
    filters: Optional[Dict] = None
) -> List[Dict]:
    """
    Search products using image query.

    Args:
        image_bytes: Raw image bytes
        token: Databricks personal access token
        num_results: Number of results to return
        filters: Optional filters

    Returns:
        List of visually similar products
    """
    # Generate image embedding
    embedding = get_image_embedding(image_bytes, token)

    # Search vector index
    vsc = VectorSearchClient()
    index = vsc.get_index(
        endpoint_name=VS_ENDPOINT,
        index_name=IMAGE_INDEX
    )

    results = index.similarity_search(
        query_vector=embedding,
        columns=[
            "product_id",
            "product_display_name",
            "master_category",
            "base_color",
            "price",
            "image_path"
        ],
        num_results=num_results,
        filters=filters
    )

    return results["result"]["data_array"]


def search_hybrid(
    query: str,
    token: str,
    num_results: int = 10,
    filters: Optional[Dict] = None
) -> List[Dict]:
    """
    Search products using hybrid embeddings (best overall quality).

    Args:
        query: Text query
        token: Databricks personal access token
        num_results: Number of results to return
        filters: Optional filters

    Returns:
        List of matching products (combines text + image signals)
    """
    # Generate text embedding (will be matched against hybrid embeddings)
    embedding = get_text_embedding(query, token)

    # Search hybrid index
    vsc = VectorSearchClient()
    index = vsc.get_index(
        endpoint_name=VS_ENDPOINT,
        index_name=HYBRID_INDEX
    )

    results = index.similarity_search(
        query_vector=embedding,
        columns=[
            "product_id",
            "product_display_name",
            "master_category",
            "article_type",
            "base_color",
            "gender",
            "season",
            "usage",
            "price",
            "image_path"
        ],
        num_results=num_results,
        filters=filters
    )

    return results["result"]["data_array"]


# Example usage
if __name__ == "__main__":
    import os

    # Get token from environment
    token = os.getenv("DATABRICKS_TOKEN")

    # Example 1: Text search
    print("=== Text Search ===")
    results = search_by_text(
        query="red leather jacket",
        token=token,
        num_results=5,
        filters={"gender": "Men"}
    )

    for i, product in enumerate(results, 1):
        print(f"{i}. {product[1]} - ${product[5]:.2f} (score: {product[-1]:.3f})")

    # Example 2: Search with price filter
    print("\n=== Budget Search ===")
    results = search_by_text(
        query="casual shirts",
        token=token,
        num_results=5,
        filters={"price <": 50}
    )

    for i, product in enumerate(results, 1):
        print(f"{i}. {product[1]} - ${product[5]:.2f}")

    # Example 3: Category-filtered search
    print("\n=== Category Search ===")
    results = search_hybrid(
        query="summer dresses",
        token=token,
        num_results=5,
        filters={"master_category": "Apparel", "season": "Summer"}
    )

    for i, product in enumerate(results, 1):
        print(f"{i}. {product[1]} - {product[5]} ({product[6]})")
