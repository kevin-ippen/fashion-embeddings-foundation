# Phases 3-5: Remaining Implementation Guides

**Note**: This file contains abbreviated versions of the remaining 3 phases. Full detailed notebooks will be provided when you're ready for each phase.

---

## Phase 3: Graph Construction

**Duration**: 4-6 hours  
**Dependencies**: Phase 2 complete  
**Deliverables**: Graph with 44K nodes, 100K-150K edges

### Quick Overview

**Objective**: Build NetworkX graph from DF2→Product mappings

**Key Steps**:
1. Load mappings from Phase 2 (~110K)
2. Create product nodes (44K) with embeddings
3. Create category nodes (5)
4. Add co-occurrence edges (from DF2 outfits)
5. Add similarity edges (high visual similarity)
6. Add category edges (product→category)
7. Persist to Delta tables

**Core Code Snippet**:
```python
import networkx as nx

# Initialize graph
G = nx.Graph()

# Add product nodes
for product in products:
    G.add_node(
        product.product_id,
        type='product',
        category=map_to_graph_category(product.article_type),
        embedding=product.image_embedding,
        price=product.price
    )

# Add edges from mappings
for outfit in df2_outfits:
    products_in_outfit = get_products_from_mapping(outfit.items)
    for p1, p2 in combinations(products_in_outfit, 2):
        if G.has_edge(p1, p2):
            G[p1][p2]['weight'] += 1
        else:
            G.add_edge(p1, p2, weight=1, edge_type='co_occurrence')

# Save graph
nx.write_gpickle(G, "/dbfs/tmp/outfit_graph.pkl")

# Save to Delta
edges_df = spark.createDataFrame([
    (u, v, G[u][v]['weight'], G[u][v].get('edge_type'))
    for u, v in G.edges()
])
edges_df.write.saveAsTable("main.fashion_demo.outfit_graph_edges")
```

**Expected Output**:
- Nodes: 44,005 (44K products + 5 categories)
- Edges: 100K-150K
- Average degree: 8-12
- Density: ~0.0001

**Tables Created**:
- `main.fashion_demo.outfit_graph_nodes`
- `main.fashion_demo.outfit_graph_edges`
- `main.fashion_demo.phase3_graph_summary`

**Full guide**: Available when you complete Phase 2

---

## Phase 4: Recommendation Engine

**Duration**: 6-8 hours  
**Dependencies**: Phase 3 complete  
**Deliverables**: Pre-computed recommendations, evaluation report

### Quick Overview

**Objective**: Compute outfit recommendations using graph

**Key Algorithms**:

#### 1. Complete the Outfit
```python
def complete_the_outfit(partial_outfit, target_category, graph, k=10):
    """
    Given 1-2 items, find best items in target_category
    
    Algorithm:
    1. Get neighbors of partial outfit items from graph
    2. Filter by target_category
    3. Score by: co_occurrence_weight + embedding_similarity
    4. Return top-k
    """
    candidates = {}
    
    for item in partial_outfit:
        neighbors = graph.neighbors(item)
        for neighbor in neighbors:
            if graph.nodes[neighbor]['category'] != target_category:
                continue
            
            # Score = edge weight + similarity
            edge_weight = graph[item][neighbor]['weight']
            similarity = cosine_sim(
                graph.nodes[item]['embedding'],
                graph.nodes[neighbor]['embedding']
            )
            
            score = 0.6 * edge_weight + 0.4 * similarity
            candidates[neighbor] = candidates.get(neighbor, 0) + score
    
    return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:k]
```

#### 2. Compatibility Scoring
```python
def compatibility_score(item1, item2, graph):
    """Rate if two items work together (0-1)"""
    
    # Co-occurrence score
    if graph.has_edge(item1, item2):
        co_occ = graph[item1][item2]['weight']
        co_occ_score = min(co_occ / 10.0, 1.0)  # Normalize
    else:
        co_occ_score = 0.0
    
    # Embedding similarity
    emb_sim = cosine_similarity(
        graph.nodes[item1]['embedding'],
        graph.nodes[item2]['embedding']
    )
    
    # Combined score
    return 0.5 * co_occ_score + 0.5 * emb_sim
```

**Pre-computation**:
```python
# For each product × each category, pre-compute top-10
for product in products:
    for category in ['tops', 'bottoms', 'shoes', 'outerwear', 'accessories']:
        if product.category == category:
            continue
        
        recommendations = complete_the_outfit([product], category, G, k=10)
        save_to_delta(product, category, recommendations)
```

**Evaluation**:
- Fill-in-blank accuracy on test outfits
- Target: 50-65%
- Coverage: % of products with recommendations
- Diversity: Average pairwise similarity in top-10

**Tables Created**:
- `main.fashion_demo.outfit_recommendations` (pre-computed)
- `main.fashion_demo.evaluation_results`
- `main.fashion_demo.phase4_summary`

**Full guide**: Available when you complete Phase 3

---

## Phase 5: API Integration & Deployment

**Duration**: 6-8 hours  
**Dependencies**: Phase 4 complete  
**Deliverables**: 3 API endpoints, production deployment

### Quick Overview

**Objective**: Deploy recommendation system to production API

**Endpoints to Add**:

#### 1. Complete the Look
```python
@router.post("/api/v1/complete-the-look")
async def complete_the_look(request: CompleteOutfitRequest):
    """
    Given 1-2 items, recommend complementary items
    
    Request:
    {
        "product_ids": [123, 456],
        "num_results": 10,
        "user_id": "optional"
    }
    
    Response:
    {
        "recommendations": [
            {
                "product_id": 789,
                "name": "Blue Sneakers",
                "category": "shoes",
                "score": 0.87,
                "image_url": "...",
                "price": 89.99
            },
            ...
        ]
    }
    """
    # Query pre-computed recommendations
    recs = await db.query("""
        SELECT recommendations, scores
        FROM main.fashion_demo.outfit_recommendations
        WHERE source_product_id = :product_id
        AND target_category NOT IN :exclude_categories
    """, product_id=request.product_ids[0], ...)
    
    # Aggregate & personalize
    final_recs = aggregate_and_personalize(recs, request.user_id)
    
    return {"recommendations": final_recs[:request.num_results]}
```

#### 2. Compatibility Score
```python
@router.post("/api/v1/compatibility-score")
async def compatibility_score(request: CompatibilityRequest):
    """
    Rate how well items work together
    
    Request:
    {
        "product_ids": [123, 456, 789]
    }
    
    Response:
    {
        "score": 0.87,
        "explanation": "Excellent combination! These items frequently appear together.",
        "pairwise_scores": [
            {"item1": 123, "item2": 456, "score": 0.91},
            {"item1": 123, "item2": 789, "score": 0.85},
            {"item1": 456, "item2": 789, "score": 0.86}
        ]
    }
    """
    # Load graph (cached in memory)
    graph = await load_cached_graph()
    
    # Compute pairwise compatibility
    scores = []
    for item1, item2 in combinations(request.product_ids, 2):
        score = compatibility_score_func(item1, item2, graph)
        scores.append({"item1": item1, "item2": item2, "score": score})
    
    overall = np.mean([s["score"] for s in scores])
    
    return {
        "score": overall,
        "explanation": generate_explanation(overall),
        "pairwise_scores": scores
    }
```

#### 3. Shop the Look
```python
@router.post("/api/v1/shop-the-look")
async def shop_the_look(image_base64: str):
    """
    Upload outfit image → find products to recreate
    
    1. Encode image with CLIP
    2. Search DF2 embeddings for similar outfits
    3. Get outfit composition from graph
    4. Map to products
    """
    # Encode image
    embedding = await clip_encoder.encode(image_base64)
    
    # Search DF2
    similar_outfits = await vector_search(
        embedding=embedding,
        index="df2_embeddings",
        k=5
    )
    
    # Get products from outfits
    all_products = []
    for outfit in similar_outfits:
        products = get_outfit_products(outfit.item_uid, graph)
        all_products.extend(products)
    
    # Rank & diversify
    return {"recommendations": rank_and_diversify(all_products)[:10]}
```

**Deployment Steps**:
1. Add endpoints to existing FastAPI app
2. Load graph on startup (pickle → memory cache)
3. Test with Postman/curl
4. Deploy to production
5. Monitor latency & errors

**Performance Targets**:
- Latency: <200ms (pre-computed queries)
- Uptime: 99%+
- Throughput: 100+ req/sec

**Full guide**: Available when you complete Phase 4

---

## 📚 Request Full Guides When Ready

**When you complete each phase**, request the full detailed guide:

**Example**:
```
"I've completed Phase 2 (mapping). Please provide the full 
GRAPH_CONSTRUCTION_GUIDE.md for Phase 3."
```

Each full guide contains:
- Complete notebook code (all cells)
- Detailed explanations
- Troubleshooting section
- Quality checks
- Expected outputs

---

## 🎯 Current Status

You should be working on:
- ✅ Phase 1: Complete (validation)
- ⏳ Phase 2: In progress (mapping) ← YOU ARE HERE
- ⏸️ Phase 3: Pending
- ⏸️ Phase 4: Pending
- ⏸️ Phase 5: Pending

**Focus**: Complete Phase 2, then request Phase 3 full guide!
