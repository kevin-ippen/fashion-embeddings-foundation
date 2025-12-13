-- ============================================================================
-- EXPORT COMPLETE TAXONOMY MAPPING
-- ============================================================================
-- Purpose: Export all article types with their categories for graph project
-- Output: Complete mapping of 143 article types to master/sub categories
-- Use: Run in Databricks SQL to create taxonomy mapping for graph construction
-- ============================================================================

-- QUERY 1: Complete Taxonomy with Counts
-- ============================================================================

SELECT
    article_type,
    master_category,
    sub_category,
    gender,
    COUNT(*) as product_count,
    MIN(price) as min_price,
    MAX(price) as max_price,
    ROUND(AVG(price), 2) as avg_price
FROM main.fashion_demo.products
WHERE article_type IS NOT NULL
GROUP BY article_type, master_category, sub_category, gender
ORDER BY master_category, sub_category, product_count DESC;

-- Expected output: ~143 article types across 7 master categories


-- QUERY 2: Master Category Summary
-- ============================================================================

SELECT
    master_category,
    COUNT(DISTINCT article_type) as num_article_types,
    COUNT(DISTINCT sub_category) as num_sub_categories,
    COUNT(*) as total_products
FROM main.fashion_demo.products
WHERE master_category IS NOT NULL
GROUP BY master_category
ORDER BY total_products DESC;

-- Expected: 7 master categories


-- QUERY 3: Article Type → Graph Category Mapping (for review)
-- ============================================================================
-- This query suggests a 5-category mapping for graph construction
-- Review and adjust as needed

SELECT
    article_type,
    master_category,
    sub_category,
    COUNT(*) as count,
    CASE
        -- TOPS
        WHEN LOWER(article_type) LIKE '%tshirt%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%shirt%' AND LOWER(article_type) NOT LIKE '%sweat%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%top%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%blouse%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%sweater%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%pullover%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%sweatshirt%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%jersey%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%tank%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%cami%' THEN 'tops'

        -- BOTTOMS
        WHEN LOWER(article_type) LIKE '%jean%' THEN 'bottoms'
        WHEN LOWER(article_type) LIKE '%trouser%' THEN 'bottoms'
        WHEN LOWER(article_type) LIKE '%pant%' THEN 'bottoms'
        WHEN LOWER(article_type) LIKE '%short%' AND master_category = 'Apparel' THEN 'bottoms'
        WHEN LOWER(article_type) LIKE '%skirt%' THEN 'bottoms'
        WHEN LOWER(article_type) LIKE '%legging%' THEN 'bottoms'
        WHEN LOWER(article_type) LIKE '%jogger%' THEN 'bottoms'
        WHEN LOWER(article_type) LIKE '%track pant%' THEN 'bottoms'

        -- SHOES
        WHEN master_category = 'Footwear' THEN 'shoes'
        WHEN LOWER(article_type) LIKE '%shoe%' THEN 'shoes'
        WHEN LOWER(article_type) LIKE '%sandal%' THEN 'shoes'
        WHEN LOWER(article_type) LIKE '%flip flop%' THEN 'shoes'
        WHEN LOWER(article_type) LIKE '%sneaker%' THEN 'shoes'
        WHEN LOWER(article_type) LIKE '%boot%' THEN 'shoes'
        WHEN LOWER(article_type) LIKE '%heel%' THEN 'shoes'

        -- OUTERWEAR
        WHEN LOWER(article_type) LIKE '%jacket%' THEN 'outerwear'
        WHEN LOWER(article_type) LIKE '%coat%' THEN 'outerwear'
        WHEN LOWER(article_type) LIKE '%blazer%' THEN 'outerwear'
        WHEN LOWER(article_type) LIKE '%cardigan%' THEN 'outerwear'
        WHEN LOWER(article_type) LIKE '%hoodie%' THEN 'outerwear'
        WHEN LOWER(article_type) LIKE '%windcheater%' THEN 'outerwear'
        WHEN LOWER(article_type) LIKE '%waistcoat%' THEN 'outerwear'
        WHEN LOWER(article_type) LIKE '%shrug%' THEN 'outerwear'

        -- DRESSES (could merge with tops or keep separate)
        WHEN LOWER(article_type) LIKE '%dress%' THEN 'dresses'
        WHEN LOWER(article_type) LIKE '%gown%' THEN 'dresses'
        WHEN LOWER(article_type) LIKE '%jumpsuit%' THEN 'dresses'

        -- ACCESSORIES
        WHEN master_category = 'Accessories' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%watch%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%bag%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%backpack%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%wallet%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%belt%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%tie%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%scarf%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%glove%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%sock%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%cap%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%hat%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%sunglasses%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%jewellery%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%jewelry%' THEN 'accessories'

        -- OTHER (non-apparel items)
        WHEN master_category IN ('Personal Care', 'Free Items', 'Home') THEN 'other'

        ELSE 'uncategorized'
    END as suggested_graph_category
FROM main.fashion_demo.products
WHERE article_type IS NOT NULL
GROUP BY article_type, master_category, sub_category
ORDER BY suggested_graph_category, count DESC;


-- QUERY 4: Check for Uncategorized Items
-- ============================================================================
-- Find article types that don't fit the 5-category mapping

WITH categorized AS (
    SELECT
        article_type,
        master_category,
        COUNT(*) as count,
        CASE
            WHEN LOWER(article_type) LIKE '%tshirt%' THEN 'tops'
            WHEN LOWER(article_type) LIKE '%shirt%' AND LOWER(article_type) NOT LIKE '%sweat%' THEN 'tops'
            WHEN LOWER(article_type) LIKE '%top%' THEN 'tops'
            WHEN LOWER(article_type) LIKE '%jean%' THEN 'bottoms'
            WHEN LOWER(article_type) LIKE '%trouser%' THEN 'bottoms'
            WHEN LOWER(article_type) LIKE '%skirt%' THEN 'bottoms'
            WHEN master_category = 'Footwear' THEN 'shoes'
            WHEN LOWER(article_type) LIKE '%jacket%' THEN 'outerwear'
            WHEN LOWER(article_type) LIKE '%coat%' THEN 'outerwear'
            WHEN LOWER(article_type) LIKE '%dress%' THEN 'dresses'
            WHEN master_category = 'Accessories' THEN 'accessories'
            WHEN master_category IN ('Personal Care', 'Free Items', 'Home') THEN 'other'
            ELSE 'uncategorized'
        END as graph_category
    FROM main.fashion_demo.products
    WHERE article_type IS NOT NULL
    GROUP BY article_type, master_category
)
SELECT *
FROM categorized
WHERE graph_category = 'uncategorized'
ORDER BY count DESC;

-- Review these and add to CASE statement above if needed


-- QUERY 5: DeepFashion2 Category Mapping
-- ============================================================================
-- Map DeepFashion2 13 categories to graph categories

SELECT
    'DeepFashion2 Mapping' as source,
    category_name as original_category,
    CASE
        WHEN category_name IN ('short_sleeve_top', 'long_sleeve_top', 'vest', 'sling') THEN 'tops'
        WHEN category_name IN ('shorts', 'trousers', 'skirt') THEN 'bottoms'
        WHEN category_name IN ('short_sleeve_outwear', 'long_sleeve_outwear') THEN 'outerwear'
        WHEN category_name IN ('short_sleeve_dress', 'long_sleeve_dress', 'vest_dress', 'sling_dress') THEN 'dresses'
        ELSE 'uncategorized'
    END as graph_category,
    COUNT(*) as item_count
FROM main.fashion_demo.df2_items
GROUP BY category_name
ORDER BY graph_category, item_count DESC;

-- Note: Check if table name is 'df2_items' or 'fashion_items_embeddings'


-- QUERY 6: Gender × Category Distribution
-- ============================================================================
-- Understand gender distribution across categories for filtering

SELECT
    master_category,
    gender,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY master_category), 1) as pct_of_category
FROM main.fashion_demo.products
WHERE master_category IS NOT NULL AND gender IS NOT NULL
GROUP BY master_category, gender
ORDER BY master_category, count DESC;


-- QUERY 7: Export Final Mapping Table
-- ============================================================================
-- Create a mapping table for use in graph construction

CREATE OR REPLACE TABLE main.fashion_demo.article_type_graph_mapping AS
SELECT DISTINCT
    article_type,
    master_category,
    sub_category,
    CASE
        -- TOPS (copy full CASE from Query 3)
        WHEN LOWER(article_type) LIKE '%tshirt%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%shirt%' AND LOWER(article_type) NOT LIKE '%sweat%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%top%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%blouse%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%sweater%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%pullover%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%sweatshirt%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%jersey%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%tank%' THEN 'tops'
        WHEN LOWER(article_type) LIKE '%cami%' THEN 'tops'

        -- BOTTOMS
        WHEN LOWER(article_type) LIKE '%jean%' THEN 'bottoms'
        WHEN LOWER(article_type) LIKE '%trouser%' THEN 'bottoms'
        WHEN LOWER(article_type) LIKE '%pant%' THEN 'bottoms'
        WHEN LOWER(article_type) LIKE '%short%' AND master_category = 'Apparel' THEN 'bottoms'
        WHEN LOWER(article_type) LIKE '%skirt%' THEN 'bottoms'
        WHEN LOWER(article_type) LIKE '%legging%' THEN 'bottoms'
        WHEN LOWER(article_type) LIKE '%jogger%' THEN 'bottoms'

        -- SHOES
        WHEN master_category = 'Footwear' THEN 'shoes'
        WHEN LOWER(article_type) LIKE '%shoe%' THEN 'shoes'
        WHEN LOWER(article_type) LIKE '%sandal%' THEN 'shoes'
        WHEN LOWER(article_type) LIKE '%flip flop%' THEN 'shoes'
        WHEN LOWER(article_type) LIKE '%sneaker%' THEN 'shoes'
        WHEN LOWER(article_type) LIKE '%boot%' THEN 'shoes'

        -- OUTERWEAR
        WHEN LOWER(article_type) LIKE '%jacket%' THEN 'outerwear'
        WHEN LOWER(article_type) LIKE '%coat%' THEN 'outerwear'
        WHEN LOWER(article_type) LIKE '%blazer%' THEN 'outerwear'
        WHEN LOWER(article_type) LIKE '%cardigan%' THEN 'outerwear'
        WHEN LOWER(article_type) LIKE '%hoodie%' THEN 'outerwear'

        -- DRESSES
        WHEN LOWER(article_type) LIKE '%dress%' THEN 'dresses'
        WHEN LOWER(article_type) LIKE '%gown%' THEN 'dresses'
        WHEN LOWER(article_type) LIKE '%jumpsuit%' THEN 'dresses'

        -- ACCESSORIES
        WHEN master_category = 'Accessories' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%watch%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%bag%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%belt%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%tie%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%cap%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%hat%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%sunglasses%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%jewellery%' THEN 'accessories'
        WHEN LOWER(article_type) LIKE '%sock%' THEN 'accessories'

        -- OTHER
        WHEN master_category IN ('Personal Care', 'Free Items', 'Home') THEN 'other'

        ELSE 'uncategorized'
    END as graph_category
FROM main.fashion_demo.products
WHERE article_type IS NOT NULL;

-- Verify the mapping table
SELECT
    graph_category,
    COUNT(DISTINCT article_type) as num_article_types,
    COUNT(*) as num_combinations
FROM main.fashion_demo.article_type_graph_mapping
GROUP BY graph_category
ORDER BY graph_category;


-- ============================================================================
-- USAGE INSTRUCTIONS
-- ============================================================================
--
-- 1. Run Query 1 first to get the complete taxonomy export
-- 2. Review Query 3 to see suggested graph category mapping
-- 3. Check Query 4 for any uncategorized items
-- 4. Adjust CASE logic as needed
-- 5. Run Query 7 to create the mapping table
-- 6. Use this table in graph construction:
--
--    SELECT p.*, m.graph_category
--    FROM main.fashion_demo.products p
--    JOIN main.fashion_demo.article_type_graph_mapping m
--      ON p.article_type = m.article_type
--
-- ============================================================================
