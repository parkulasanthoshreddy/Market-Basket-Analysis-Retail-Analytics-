**Market Basket Analysis (Retail Analytics)**

This project analyzes retail transaction data to find which products are often bought together and to understand customer purchasing behavior.
It uses the Apriori algorithm for association rule mining and RFM analysis for customer segmentation.

**1. Project Summary**

The goal of this project is to:

Identify frequently bought-together items

Generate association rules (e.g., If a customer buys X, they might also buy Y) 

Segment customers using RFM (Recency, Frequency, Monetary)

Help retailers understand buying patterns and customer value

This project is built with Python and uses libraries like Pandas, mlxtend, Matplotlib, and Seaborn.

**2. Dataset**

File: dataset/orders.csv

Total rows: 38,765

Columns:

Member_number → Customer ID

Date → Purchase date

itemDescription → Product purchased

Each row represents one product bought by one customer.

**3. What the Project Does**

Step 1: Data Cleaning

Renames columns to standard names

Converts dates

Cleans item names

Creates a unique InvoiceNo for each purchase day

Adds Quantity = 1 because dataset doesn’t include quantity

Step 2: Market Basket Matrix

Creates a table showing which items appear in each transaction (True/False values).

Step 3: Apriori Algorithm

Finds:

Frequent items

Frequent item combinations

Association rules with confidence and lift

Saved to:

frequent_itemsets.csv

association_rules.csv

Step 4: RFM Segmentation

Calculates:

Recency → how recently a customer purchased

Frequency → how often they buy

Monetary → total items bought

Saved to:

rfm_segments.csv

Step 5: Visualizations

Project generates:

recency_dist.png

frequency_dist.png

These help understand shopping frequency and recency patterns.

**4. Outputs Folder**

All results are stored in:

outputs/


Files include:

Frequent Itemsets

Association Rules

RFM Segments

Recency & Frequency Charts

**5. How to Run the Project**

Open the project folder

Create and activate a virtual environment:

python -m venv .venv
.\.venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Run the project:

python mba_starter.py


All results will appear inside the outputs/ folder.
