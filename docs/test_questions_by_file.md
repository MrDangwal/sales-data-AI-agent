# Test Question Bank (5 per File)

## 1) Amazon Sale Report.csv (`sales_amazon_sale_report` / `mart_amazon_orders`)
1. Top 10 states by total sales and total units.
2. Cancellation rate by fulfillment channel with total order counts.
3. Month-wise total sales trend with total orders.
4. Top 10 categories by average order value and total sales.
5. Top 15 cities by revenue contribution.

## 2) International sale Report.csv (`sales_international_sale_report`)
1. Top 10 customers by gross amount.
2. Month-wise gross amount trend.
3. Top styles by total gross amount and pieces sold.
4. Size-wise contribution to gross amount.
5. Highest average rate SKUs with at least 20 records.

## 3) Sale Report.csv (`sales_sale_report`)
1. Top categories by available stock.
2. Color-wise stock distribution.
3. Size-wise stock concentration within each category.
4. Top 20 SKU codes by stock.
5. Designs with widest size availability.

## 4) May-2022.csv (`sales_may_2022`)
1. Compare channel MRP columns and find highest priced channel per SKU.
2. Top categories by average TP and average MRP.
3. SKUs with biggest spread between TP and Amazon MRP.
4. Category-wise average weight and TP.
5. Styles with most SKUs and their average channel MRP.

## 5) P  L March 2021.csv (`sales_p_l_march_2021`)
1. Difference analysis between TP 1 and TP 2 by category.
2. Top categories by average margin proxy (MRP Old - TP 2).
3. SKUs with highest TP 2 values.
4. Channel MRP consistency check across Ajio/Amazon/Flipkart/Myntra.
5. Category and size combinations with highest Final MRP Old.

## 6) Expense IIGF.csv (`sales_expense_iigf`)
1. Total received amount vs total expense amount.
2. Top expense particulars by amount.
3. Net balance after expenses.
4. Expense share (%) by expense particular.
5. Date-level incoming amount trend.

## 7) Cloud Warehouse Compersion Chart.csv (`sales_cloud_warehouse_compersion_chart`)
1. Head-wise price comparison between Shiprocket and INCREFF.
2. Average unit price difference by provider.
3. Heads where Shiprocket is cheaper vs INCREFF.
4. Heads where INCREFF is cheaper vs Shiprocket.
5. Overall cost delta if all heads were procured from one provider.

## Cross-File Validation Questions
1. Compare stock-heavy categories (`sales_sale_report`) with high-revenue categories (`mart_amazon_orders`).
2. Compare domestic sales trend with international gross amount trend.
3. Evaluate if higher MRP channels align with higher sales contribution.
4. Cross-check pricing spread (May/P&L files) against cancellation-heavy states.
5. Build a combined executive summary from all datasets.
