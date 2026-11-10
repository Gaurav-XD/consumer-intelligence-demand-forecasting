-- Consumer analytics examples for PostgreSQL, one independent SELECT per section.
-- Execute a chosen statement using a client that binds $1 as an inclusive DATE.
-- The tables customers, transactions, products and marketing must already exist
-- with the names and types in src/consumer_intelligence/data_schema.py.
-- transaction_id is assumed unique, one transaction is one order, and dates
-- must be complete through the supplied cutoff. No data is created or modified.
-- Revenue uses exact PostgreSQL numeric arithmetic and per-transaction rounding;
-- floating-point tie rounding may differ by one cent from the pandas pipeline.

-- 1. CUSTOMER REVENUE AND ORDER METRICS
-- Include customers without orders and exclude customers not yet signed up.
WITH parameters AS (
    SELECT $1::date AS snapshot_date
),
orders AS (
    SELECT t.customer_id, t.transaction_id, t.quantity,
           ROUND(t.quantity * t.price::numeric * (1 - t.discount::numeric), 2) AS revenue
    FROM transactions AS t
    CROSS JOIN parameters AS p
    WHERE t.transaction_date <= p.snapshot_date
)
SELECT c.customer_id,
       COUNT(o.transaction_id) AS total_orders,
       COALESCE(SUM(o.quantity), 0) AS total_quantity,
       COALESCE(SUM(o.revenue), 0::numeric) AS total_revenue,
       ROUND(AVG(o.revenue), 2) AS average_order_value
FROM customers AS c
CROSS JOIN parameters AS p
LEFT JOIN orders AS o ON o.customer_id = c.customer_id
WHERE c.signup_date <= p.snapshot_date
GROUP BY c.customer_id
ORDER BY c.customer_id;

-- 2. RFM-STYLE CUSTOMER METRICS
-- Recency is undefined for customers with no purchase; frequency is order count.
WITH parameters AS (
    SELECT $1::date AS snapshot_date
),
historical_orders AS (
    SELECT t.customer_id, t.transaction_id, t.transaction_date,
           ROUND(t.quantity * t.price::numeric * (1 - t.discount::numeric), 2) AS revenue
    FROM transactions AS t
    CROSS JOIN parameters AS p
    WHERE t.transaction_date <= p.snapshot_date
)
SELECT c.customer_id,
       p.snapshot_date - MAX(o.transaction_date) AS recency_days,
       COUNT(o.transaction_id) AS frequency_orders,
       COALESCE(SUM(o.revenue), 0::numeric) AS monetary_total_revenue
FROM customers AS c
CROSS JOIN parameters AS p
LEFT JOIN historical_orders AS o ON o.customer_id = c.customer_id
WHERE c.signup_date <= p.snapshot_date
GROUP BY c.customer_id, p.snapshot_date
ORDER BY c.customer_id;

-- 3. CUSTOMER SEGMENTATION INPUTS
-- Tie-aware cumulative distribution percentiles are exploratory inputs, not
-- the pandas RFM scores or segment labels; zero-history customers remain visible.
WITH parameters AS (
    SELECT $1::date AS snapshot_date
),
customer_metrics AS (
    SELECT c.customer_id, c.region,
           COUNT(t.transaction_id) AS frequency_orders,
           COALESCE(SUM(ROUND(t.quantity * t.price::numeric * (1 - t.discount::numeric), 2)), 0::numeric) AS monetary_value,
           p.snapshot_date - MAX(t.transaction_date) AS recency_days
    FROM customers AS c
    CROSS JOIN parameters AS p
    LEFT JOIN transactions AS t
      ON t.customer_id = c.customer_id AND t.transaction_date <= p.snapshot_date
    WHERE c.signup_date <= p.snapshot_date
    GROUP BY c.customer_id, c.region, p.snapshot_date
),
ranked AS (
    SELECT customer_metrics.*,
           CUME_DIST() OVER (ORDER BY monetary_value) AS monetary_percentile,
           CUME_DIST() OVER (ORDER BY frequency_orders) AS frequency_percentile
    FROM customer_metrics
)
SELECT customer_id, region, recency_days, frequency_orders, monetary_value,
       monetary_percentile, frequency_percentile,
       CASE WHEN frequency_orders = 0 THEN 'No purchase history'
            WHEN recency_days <= 90 THEN 'Recent purchaser'
            ELSE 'Less recent purchaser'
       END AS activity_group
FROM ranked
ORDER BY customer_id;

-- 4. MONTHLY REVENUE AND DEMAND TRENDS
-- A month is based on transaction date; the final month may be partial.
WITH monthly AS (
    SELECT DATE_TRUNC('month', t.transaction_date)::date AS month,
           COUNT(*) AS order_count,
           SUM(t.quantity) AS quantity_sold,
           SUM(ROUND(t.quantity * t.price::numeric * (1 - t.discount::numeric), 2)) AS revenue
    FROM transactions AS t
    WHERE t.transaction_date <= $1::date
    GROUP BY DATE_TRUNC('month', t.transaction_date)::date
)
SELECT month, order_count, quantity_sold, revenue,
       LAG(revenue) OVER (ORDER BY month) AS previous_observed_month_revenue
FROM monthly
ORDER BY month;

-- 5. COHORT AND RETENTION ANALYSIS
-- Cohorts are first observed purchase month. Include complete months M0-M12;
-- matured months without purchases are zero and unobserved months are omitted.
WITH parameters AS (
    SELECT $1::date AS snapshot_date
),
coverage AS (
    SELECT CASE
             WHEN snapshot_date = (DATE_TRUNC('month', snapshot_date) + INTERVAL '1 month' - INTERVAL '1 day')::date
             THEN DATE_TRUNC('month', snapshot_date)::date
             ELSE (DATE_TRUNC('month', snapshot_date) - INTERVAL '1 month')::date
           END AS last_complete_month
    FROM parameters
),
history AS (
    SELECT t.customer_id,
           DATE_TRUNC('month', t.transaction_date)::date AS month,
           ROUND(t.quantity * t.price::numeric * (1 - t.discount::numeric), 2) AS revenue
    FROM transactions AS t
    CROSS JOIN coverage AS c
    WHERE t.transaction_date < (c.last_complete_month + INTERVAL '1 month')::date
),
first_purchase AS (
    SELECT customer_id, MIN(month) AS cohort_month
    FROM history
    GROUP BY customer_id
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(*) AS cohort_size
    FROM first_purchase
    GROUP BY cohort_month
),
cohort_activity AS (
    SELECT f.cohort_month, h.month, COUNT(DISTINCT h.customer_id) AS retained_customers,
           SUM(h.revenue) AS cohort_revenue
    FROM history AS h
    JOIN first_purchase AS f ON f.customer_id = h.customer_id
    GROUP BY f.cohort_month, h.month
),
cohort_grid AS (
    SELECT s.cohort_month, s.cohort_size, period.n AS period,
           (s.cohort_month + period.n * INTERVAL '1 month')::date AS month
    FROM cohort_sizes AS s
    CROSS JOIN generate_series(0, 12) AS period(n)
    CROSS JOIN coverage AS c
    WHERE (s.cohort_month + period.n * INTERVAL '1 month')::date <= c.last_complete_month
),
cohort_values AS (
    SELECT g.cohort_month, g.month, g.period, g.cohort_size,
           COALESCE(a.retained_customers, 0) AS retained_customers,
           COALESCE(a.cohort_revenue, 0::numeric) AS cohort_revenue
    FROM cohort_grid AS g
    LEFT JOIN cohort_activity AS a ON a.cohort_month = g.cohort_month AND a.month = g.month
),
with_baseline AS (
    SELECT cohort_values.*,
           FIRST_VALUE(cohort_revenue) OVER (PARTITION BY cohort_month ORDER BY period) AS m0_revenue
    FROM cohort_values
)
SELECT cohort_month, month, period, cohort_size, retained_customers,
       retained_customers::numeric / NULLIF(cohort_size, 0) AS retention_rate,
       cohort_revenue, m0_revenue,
       cohort_revenue / NULLIF(m0_revenue, 0) AS revenue_retention
FROM with_baseline
ORDER BY cohort_month, period;

-- 6. CHURN FEATURE PREPARATION
-- All features use only data available through the snapshot. No future-derived
-- label is included; define and validate a separate observation window later.
WITH parameters AS (
    SELECT $1::date AS snapshot_date
),
customer_orders AS (
    SELECT t.customer_id,
           COUNT(*) AS total_orders,
           SUM(t.quantity) AS total_quantity,
           SUM(ROUND(t.quantity * t.price::numeric * (1 - t.discount::numeric), 2)) AS total_revenue,
           ROUND(AVG(ROUND(t.quantity * t.price::numeric * (1 - t.discount::numeric), 2)), 2) AS average_order_value,
           AVG(t.discount) AS average_discount,
           COUNT(DISTINCT t.product_id) AS unique_products_purchased,
           COUNT(DISTINCT p.category) AS unique_categories_purchased,
           MIN(t.transaction_date) AS first_purchase_date,
           MAX(t.transaction_date) AS last_purchase_date
    FROM transactions AS t
    LEFT JOIN products AS p ON p.product_id = t.product_id
    CROSS JOIN parameters AS cutoff
    WHERE t.transaction_date <= cutoff.snapshot_date
    GROUP BY t.customer_id
),
customer_marketing AS (
    SELECT m.customer_id, COUNT(*) AS number_marketing_interactions,
           COUNT(DISTINCT m.campaign) AS number_marketing_campaigns
    FROM marketing AS m
    CROSS JOIN parameters AS p
    WHERE m.campaign_date <= p.snapshot_date
    GROUP BY m.customer_id
)
SELECT c.customer_id, p.snapshot_date,
       p.snapshot_date - c.signup_date AS customer_tenure_days,
       COALESCE(o.total_orders, 0) AS total_orders,
       COALESCE(o.total_quantity, 0) AS total_quantity,
       COALESCE(o.total_revenue, 0::numeric) AS total_revenue,
       o.average_order_value, o.average_discount,
       COALESCE(o.unique_products_purchased, 0) AS unique_products_purchased,
       COALESCE(o.unique_categories_purchased, 0) AS unique_categories_purchased,
       CASE WHEN p.snapshot_date > c.signup_date
            THEN COALESCE(o.total_orders, 0) * 365.25 / (p.snapshot_date - c.signup_date)
       END AS purchase_frequency,
       (o.last_purchase_date - o.first_purchase_date)::numeric / NULLIF(o.total_orders - 1, 0) AS average_days_between_purchases,
       p.snapshot_date - o.last_purchase_date AS days_since_last_purchase,
       o.last_purchase_date - o.first_purchase_date AS customer_lifetime_days,
       COALESCE(m.number_marketing_interactions, 0) AS number_marketing_interactions,
       COALESCE(m.number_marketing_campaigns, 0) AS number_marketing_campaigns
FROM customers AS c
CROSS JOIN parameters AS p
LEFT JOIN customer_orders AS o ON o.customer_id = c.customer_id
LEFT JOIN customer_marketing AS m ON m.customer_id = c.customer_id
WHERE c.signup_date <= p.snapshot_date
ORDER BY c.customer_id;

-- 7. PRODUCT AND CATEGORY PERFORMANCE
-- Preserve products with no sales and rank products within their category.
WITH product_totals AS (
    SELECT p.product_id, p.category, p.brand,
           COUNT(t.transaction_id) AS order_count,
           COALESCE(SUM(t.quantity), 0) AS quantity_sold,
           COALESCE(SUM(ROUND(t.quantity * t.price::numeric * (1 - t.discount::numeric), 2)), 0::numeric) AS revenue
    FROM products AS p
    LEFT JOIN transactions AS t ON t.product_id = p.product_id AND t.transaction_date <= $1::date
    GROUP BY p.product_id, p.category, p.brand
)
SELECT product_id, category, brand, order_count, quantity_sold, revenue,
       SUM(revenue) OVER (PARTITION BY category) AS category_revenue,
       DENSE_RANK() OVER (PARTITION BY category ORDER BY revenue DESC) AS category_revenue_rank
FROM product_totals
ORDER BY category, category_revenue_rank, product_id;
