# E-Commerce Customer Analytics

Analyzing customer behavior and retention patterns using data from Olist, a Brazilian e-commerce platform. This project helped me understand how businesses track customer value and predict repeat purchases.

## What This Project Does

I worked with about 100,000 orders to figure out:
- Who are the most valuable customers and what makes them different
- Whether happy customers actually come back more often
- How shopping patterns vary across different regions in Brazil
- Which customers stick around vs. those who buy once and disappear

## The Analysis

**Revenue Trends**  
Looked at monthly revenue over time to spot any seasonal patterns or growth trends.

**Customer Segmentation**  
Split customers into four groups based on how much they spend (Low, Medium, High, VIP). This helps identify which segment drives most of the revenue.

**Payment Preferences**  
Mapped out which payment methods people prefer - credit cards, bank transfers, etc. Interesting to see regional differences here.

**Geographic Breakdown**  
São Paulo and Rio dominate (as expected), but wanted to see the actual numbers and if smaller states punch above their weight.

**Satisfaction vs. Loyalty**  
The main question: do higher review scores actually correlate with repeat purchases? Tested this with customers who bought multiple times.

**Cohort Retention**  
Tracked customers by their first purchase month to see how many came back in Month 2, Month 3, and so on. This shows where the drop-off happens.

## Tech Used

- Python for everything
- Pandas for wrangling the data
- Matplotlib/Seaborn for charts
- Numpy for the correlation calculations

## How to Run This

First, grab the dataset from Kaggle:  
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Put these files in the project folder:
- olist_orders_dataset.csv
- olist_order_payments_dataset.csv
- olist_customers_dataset.csv
- olist_order_reviews_dataset.csv

Then install what you need:
```bash
pip install pandas matplotlib seaborn numpy
```

Run it:
```bash
python ecommerce_analysis.py
```

You'll get six charts saved as PNGs plus some summary stats in the console.

## What I Learned

The biggest thing was understanding cohort analysis - it's way more useful than just looking at overall retention rates. You can actually see which months acquired the stickiest customers.

Also found it interesting how the quartile-based segmentation revealed a small group of VIP customers contributing a disproportionate amount of revenue. Classic Pareto principle in action.

## Files

```
ecommerce_analysis.py  - main script with all the analysis
README.md              - this file
requirements.txt       - dependencies
.gitignore            - keeps CSV files out of git
DATA.md               - dataset info
```

## Next Steps

Some things I want to add:
- RFM analysis (Recency, Frequency, Monetary value)
- Look at product categories if I bring in that dataset
- Maybe build a simple churn prediction model
- See if delivery time affects satisfaction scores

## Notes

The CSV files aren't in this repo because they're pretty big (~50MB). Download them from the Kaggle link above.

Dataset is under CC BY-NC-SA 4.0 license - free to use for learning and analysis.
