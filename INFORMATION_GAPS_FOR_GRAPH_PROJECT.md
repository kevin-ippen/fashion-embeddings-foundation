# Information Gaps - Graph-Based Outfit Composition Project

**Date**: 2025-12-13
**Purpose**: Identify missing information needed to build Polyvore-style graph + embeddings system
**Target**: Claude Code + Databricks implementation

---

## 🎯 Project Goal (Based on Conversation)

Build a **graph-based outfit composition system** that:
- Maps 33K DeepFashion2 images → 44K product catalog
- Creates co-occurrence graph from outfit images
- Enables "shop the look" and "complete the outfit" features
- Combines visual embeddings + graph structure (better than embeddings alone)

---

## ❗ CRITICAL GAPS - Need Immediate Answers

### 1. DeepFashion2 Dataset Characteristics

**What we need to know:**

```
Q1: What do the 33K DeepFashion2 images actually contain?
   [ ] Individual product images (like product catalog)
   [ ] Full outfit images with people wearing multiple items
   [ ] Mix of both
   [ ] Other: _______________

Q2: If outfit images, what's the typical composition?
   - Average number of visible items per image: _____
   - Types of items typically visible: _____
   - Photography style:
     [ ] Clean studio shots
     [ ] Street style / lifestyle photography
     [ ] Fashion runway
     [ ] Mix of styles
     [ ] Other: _______________

Q3: What metadata exists for DeepFashion2 images?
   [ ] Bounding boxes for individual items
   [ ] Segmentation masks
   [ ] Category labels per item
   [ ] Keypoints / pose information
   [ ] Item-level attributes (color, style, etc.)
   [ ] Outfit-level tags
   [ ] None of the above
   [ ] Other: _______________

Q4: Where are these images stored and how can they be accessed?
   - Storage path: _____________________
   - Format: [ ] DBFS [ ] Unity Catalog Volumes [ ] S3 [ ] Other: _____
   - Average file size: _____
   - Can we load them in notebooks? [ ] Yes [ ] No [ ] Unknown
```

**Why this matters**: 
- If outfit images → Can build co-occurrence graph directly
- If individual items → Need different approach
- If has bounding boxes → Can extract individual garments
- Metadata determines what features we can build

---

### 2. Category Taxonomy Details

**What we need:**

```
Q5: Complete mapping of 143 article types to major categories

Please provide a CSV or list like:
article_type,major_category,graph_category
"Shirts","Apparel","tops"
"Jeans","Apparel","bottoms"
"Casual Shoes","Footwear","shoes"
...

Or provide access to a query that returns this:
SELECT article_type, master_category, sub_category 
FROM main.fashion_demo.products 
GROUP BY article_type, master_category, sub_category

We need to map to these 5 graph categories:
- tops (shirts, t-shirts, blouses, sweaters, etc.)
- bottoms (jeans, pants, skirts, shorts, etc.)
- shoes (all footwear)
- outerwear (jackets, coats, etc.)
- accessories (bags, watches, jewelry, etc.)

Q6: Are there article types that don't fit these categories?
   List any: _____________________
   How should we handle them? _____________________
```

**Why this matters**: Graph construction requires category nodes (core to Polyvore approach)

---

### 3. Existing Product-DeepFashion2 Relationships

**What we need:**

```
Q7: Has any analysis been done on similarity between datasets?

   [ ] Yes, we've mapped similar items across datasets
   [ ] We've run some similarity searches but not systematically
   [ ] No mapping has been attempted
   [ ] Not sure

Q8: If yes to Q7, where is this data stored?
   - Table/path: _____________________
   - What was the approach: _____________________

Q9: Do the DeepFashion2 images and product catalog have any overlap?
   [ ] Yes, same items appear in both
   [ ] No, completely different items
   [ ] Some overlap but mostly different
   [ ] Unknown
```

**Why this matters**: Determines if we can create training data for outfit composition

---

### 4. Infrastructure & Access Details

**What we need:**

```
Q10: Databricks workspace details
   - Workspace URL: _____________________
   - Workspace region: _____________________
   - Unity Catalog name: main (confirmed)
   - Schema: fashion_demo (confirmed)

Q11: Compute resources available
   Current cluster type for ML work:
   [ ] Single-node (driver only)
   [ ] Multi-node
   - Driver node type: _____________________
   - Worker node type: _____________________
   - Number of workers: _____
   
   GPU availability:
   [ ] No GPU
   [ ] Single GPU (type: _____)
   [ ] Multi-GPU (type: _____, count: _____)
   
   Preferred cluster for new project:
   [ ] Reuse existing
   [ ] Create new ML cluster
   - Preferred node type: _____________________
   - DBR version preference: _____________________

Q12: Storage configuration
   Where do you want new data stored?
   - Delta tables: main.fashion_demo._____ (preferred naming?)
   - Volumes path for graphs/arrays: _____________________
   - DBFS path (if used): _____________________

Q13: Python package installation
   Current approach:
   [ ] %pip install in notebooks
   [ ] Cluster libraries
   [ ] Both
   - Any restrictions on packages? _____________________
```

**Why this matters**: Determines implementation approach and performance

---

## 📊 IMPORTANT GAPS - Need for Optimization

### 5. Current Embedding Access Patterns

**What we need:**

```
Q14: How do you currently access embeddings in production?

   For products (44K items):
   [ ] Query Delta table and parse ARRAY<DOUBLE> in Python
   [ ] Pre-loaded numpy arrays in memory
   [ ] Vector Search API only
   [ ] Other: _____________________

Q15: Are embeddings also stored as files (npy, parquet)?
   [ ] Yes, in _____________________
   [ ] No, only in Delta tables
   [ ] Not sure

Q16: If we need to load all 44K product embeddings into memory:
   - What's the current approach? _____________________
   - How long does it take? _____________________
   - Any memory issues? [ ] Yes [ ] No [ ] Haven't tried
```

**Why this matters**: Graph algorithms need fast embedding access (numpy arrays preferred)

---

### 6. Model Upgrade Path

**What we need:**

```
Q17: Interest in upgrading from CLIP ViT-B/32?

   Priority for this project:
   [ ] High - Want to use FashionSigLIP or SigLIP 2
   [ ] Medium - Open to it if time permits
   [ ] Low - Stick with existing CLIP embeddings
   [ ] Not interested - Use what we have

Q18: If upgrading, what's the preference?
   [ ] Marqo-FashionSigLIP (best for fashion, +57% improvement)
   [ ] FashionCLIP 2.0 (proven, widely used)
   [ ] SigLIP 2 base + fine-tune (most sophisticated)
   [ ] Keep CLIP ViT-B/32 (focus on graph approach)

Q19: Re-embedding constraints
   - Budget for GPU compute: $_____ (or N/A if flexible)
   - Timeline tolerance for re-embedding: _____ days
   - Can we do it incrementally? [ ] Yes [ ] No [ ] Either way
```

**Why this matters**: Better embeddings = 2-3x better results, but requires re-processing

---

### 7. Use Case Prioritization

**What we need:**

```
Q20: Rank these features by priority (1 = highest, 5 = lowest)

   ___ "Shop the look": Given outfit image, find products to recreate it
   ___ "Complete the outfit": Given 1-2 items, suggest complementary items
   ___ Outfit compatibility scoring: Rate if items go well together
   ___ Style-based recommendations: "More like this but different category"
   ___ Trend discovery: Identify which items frequently appear together

Q21: Primary user interaction model
   [ ] User uploads outfit image → get product recommendations
   [ ] User selects products → get completion suggestions
   [ ] User browses → personalized "you might also like"
   [ ] Research/analytics (not user-facing initially)
   [ ] Other: _____________________

Q22: Initial scope preference
   [ ] Start small (1000 products, proof of concept)
   [ ] Medium scope (10K products, MVP)
   [ ] Full scale (all 44K products, production-ready)
   [ ] Flexible - Claude decides based on complexity
```

**Why this matters**: Determines which graph algorithms to prioritize

---

### 8. Graph Infrastructure

**What we need:**

```
Q23: Do you have any existing graph infrastructure?
   [ ] Neo4j or other graph database
   [ ] NetworkX or graph libraries used before
   [ ] None currently
   [ ] Not sure

Q24: Preference for graph storage
   [ ] In-memory NetworkX graphs (simpler, good for <100K nodes)
   [ ] Delta tables with edge/node tables (scalable, queryable)
   [ ] Dedicated graph database
   [ ] No preference - Claude decides

Q25: Any existing co-occurrence or compatibility data?
   [ ] Yes, in _____________________
   [ ] No, need to build from scratch
   [ ] Partially - explain: _____________________
```

**Why this matters**: Determines if we build from scratch or augment existing

---

## 🔍 HELPFUL CONTEXT - Nice to Have

### 9. Real-World Data Availability

**What we have:**

```
Q26: Any real user interaction data (even small samples)?
   [ ] Yes - _____ interactions available
       Location: _____________________
   [ ] No, only synthetic data
   [ ] Coming soon

Q27: Any manually curated outfit examples?
   [ ] Yes - _____ outfits curated
       Location: _____________________
   [ ] No
   [ ] Could create some

Q28: Any domain expert input available?
   [ ] Fashion expert can validate results
   [ ] Internal team can review
   [ ] No expert validation available
   [ ] Other: _____________________
```

**Why this matters**: Real data = better validation, expert input = better graph rules

---

### 10. Performance & Scale Requirements

**What we need:**

```
Q29: Expected production scale
   - Queries per second: _____ (or "not applicable yet")
   - Latency requirement: _____ ms
   - Is this for: [ ] Research [ ] MVP [ ] Production
   
Q30: Acceptable computation time
   For building initial graph:
   [ ] Minutes (small-scale testing)
   [ ] Hours (batch processing overnight)
   [ ] Days (one-time computation)
   [ ] No preference

Q31: Online vs. Batch processing
   [ ] Need real-time recommendations (<100ms)
   [ ] Batch recommendations are fine (nightly processing)
   [ ] Hybrid (pre-compute some, real-time for others)
```

**Why this matters**: Determines architecture (pre-computed vs. on-demand)

---

## 📋 Data Collection Checklist

Please gather the following and provide in next message:

### Required (Can't proceed without):
- [ ] Q1-Q4: DeepFashion2 dataset characteristics
- [ ] Q5-Q6: Complete category taxonomy mapping
- [ ] Q10-Q12: Infrastructure access details

### High Priority (Needed soon):
- [ ] Q7-Q9: Product-DeepFashion2 relationships
- [ ] Q14-Q16: Embedding access patterns
- [ ] Q17-Q19: Model upgrade preferences
- [ ] Q20-Q22: Use case prioritization

### Medium Priority (Helpful for planning):
- [ ] Q23-Q25: Graph infrastructure preferences
- [ ] Q26-Q28: Real-world data availability

### Low Priority (Can assume defaults):
- [ ] Q29-Q31: Performance requirements

---

## 🎯 What We'll Build Once We Have This Info

Based on your answers, we'll create:

1. **Project specification MD** with:
   - Detailed architecture
   - Data pipeline design
   - Graph construction approach
   - Model serving strategy
   - Evaluation metrics

2. **Claude Code project structure** with:
   - Databricks notebook workflow
   - Python package for graph utilities
   - Vector search integration
   - API endpoints (if needed)
   - Testing framework

3. **Implementation plan** with:
   - Step-by-step development phases
   - Estimated timelines
   - Resource requirements
   - Success metrics

---

## 📝 How to Provide This Information

**Option 1: Structured responses**
Copy this document and fill in answers directly

**Option 2: Narrative format**
Just tell me about your setup conversationally and I'll extract answers

**Option 3: Show me your data**
Share sample queries, screenshots, or code snippets that reveal the answers

**Option 4: Mixed approach**
Answer what you know, flag what you need to investigate

---

## ⚡ Quick Win Suggestion

While you gather this info, I can already start on:

```python
# Minimal working example with assumptions
# We can refine once you provide details

# Assumptions:
# - DeepFashion2 = outfit images (not individual items)
# - Has basic category labels
# - Can map 143 article types to 5 major categories
# - Use existing CLIP embeddings
# - Build in-memory NetworkX graph for MVP

# Then we'll scale/refine based on your answers
```

Should I proceed with this quick-win approach while you gather details? 
Or wait for complete information first?

---

**Next Steps:**
1. Review these questions
2. Gather information you have readily available
3. Flag questions that need investigation
4. Provide answers in your preferred format
5. I'll create comprehensive project spec + implementation plan

Let me know what information you already have and what you need to look up! 🚀
