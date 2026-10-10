-- Retail customer churn analysis
-- Tables: customers (from customers_clean.csv), orders (from orders_clean.csv)

-- 1. Churn rate by membership tier
SELECT membership_tier,
       COUNT(*)                                   AS customers,
       ROUND(100.0 * AVG(churned_flag), 1)        AS churn_rate_pct
FROM customers
GROUP BY membership_tier
ORDER BY churn_rate_pct DESC;

-- 2. Customer spend and order count (JOIN + aggregates)
SELECT c.customer_id,
       c.membership_tier,
       c.churned_flag,
       COUNT(o.order_id)              AS order_count,
       ROUND(SUM(o.net_amount_inr),2) AS total_spend_inr
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id
GROUP BY c.customer_id, c.membership_tier, c.churned_flag;

-- 3. Return rate for churned vs active customers
SELECT c.churned,
       COUNT(o.order_id)                                        AS orders,
       SUM(CASE WHEN o.order_status = 'Returned' THEN 1 ELSE 0 END) AS returned,
       ROUND(100.0 * SUM(CASE WHEN o.order_status = 'Returned' THEN 1 ELSE 0 END)
             / COUNT(o.order_id), 1)                            AS return_rate_pct
FROM customers c
JOIN orders o ON o.customer_id = c.customer_id
GROUP BY c.churned;

-- 4. Running order count per customer (window function)
SELECT o.customer_id,
       o.order_date,
       o.net_amount_inr,
       COUNT(*) OVER (PARTITION BY o.customer_id ORDER BY o.order_date
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_order_count
FROM orders o
WHERE o.customer_id IN ('C100000','C100001','C100002')
ORDER BY o.customer_id, o.order_date;

-- 5. Top 10 customers by spend
SELECT o.customer_id,
       ROUND(SUM(o.net_amount_inr),2) AS total_spend_inr
FROM orders o
GROUP BY o.customer_id
ORDER BY total_spend_inr DESC
LIMIT 10;
