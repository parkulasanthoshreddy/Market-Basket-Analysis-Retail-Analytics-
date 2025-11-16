import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules


# 1. PATH CONFIGURATION

PROJECT_DIR = Path(
    r"C:\Users\Santhosh\OneDrive\Desktop\projects\Market Basket Analysis (Retail Analytics)"
)

DATA_FILE = PROJECT_DIR / "dataset" / "orders.csv"
OUT_DIR = PROJECT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. LOAD & CLEAN DATA

def load_orders(path: Path) -> pd.DataFrame:
    """Load the orders CSV and do minimal validation."""
    print(f" Loading data from: {path}")
    df = pd.read_csv(path, encoding="utf-8")

    print("\n Raw data preview:")
    print(df.head())

    print("\n  Data info:")
    print(df.info())

    return df


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, clean text, create InvoiceNo and Quantity."""
    df = df.copy()

    # Rename columns to more standard names
    df.rename(
        columns={
            "Member_number": "CustomerID",
            "itemDescription": "Description",
            "Date": "InvoiceDate",
        },
        inplace=True,
    )

    # Ensure basic columns exist
    required_cols = ["CustomerID", "Description", "InvoiceDate"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Clean text
    df["Description"] = (
        df["Description"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Parse dates (dataset is dd-mm-yyyy)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    # Create a synthetic invoice id: one basket per (customer, day)
    df["InvoiceNo"] = (
        df["CustomerID"].astype(str) + "_" + df["InvoiceDate"].dt.strftime("%Y-%m-%d")
    )

    # Each row represents one purchased product -> Quantity = 1
    df["Quantity"] = 1

    print("\n Cleaned data preview:")
    print(df.head())

    return df


# 3. MARKET BASKET ANALYSIS (APRIORI)

def build_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a binary basket matrix (InvoiceNo x Description)."""
    print("\n Building basket matrix (InvoiceNo x Description)...")

    basket = (
        df.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )

    # Convert counts to True/False for Apriori (vectorized)
    basket_binary = basket > 0
    basket_binary = basket_binary.astype(bool)

    print(f"Basket shape: {basket_binary.shape}")
    return basket_binary


def run_apriori_and_rules(basket_sets: pd.DataFrame,
                          min_support: float = 0.005) -> pd.DataFrame:
    """
    Run Apriori to find frequent itemsets and derive association rules.
    Uses lower support & confidence so we actually see patterns.
    """
    print(f"\n Running Apriori with min_support={min_support}...")

    basket_sets = basket_sets.astype(bool)

    frequent_itemsets = apriori(
        basket_sets,
        min_support=min_support,
        use_colnames=True
    )

    frequent_itemsets["itemset_len"] = frequent_itemsets["itemsets"].apply(len)
    frequent_itemsets.sort_values("support", ascending=False, inplace=True)
    frequent_itemsets.to_csv(OUT_DIR / "frequent_itemsets.csv", index=False)
    print(" Saved frequent_itemsets.csv")

    print("\nTop frequent itemsets:")
    print(frequent_itemsets.head())

    # How many 2+ item sets did we get?
    print("\nNumber of itemsets with length >= 2:",
          (frequent_itemsets["itemset_len"] >= 2).sum())

    print("\n Generating association rules (metric='confidence', min_threshold=0.1)...")
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.1  # 10% confidence
    )

    if rules.empty:
        print("\nNo association rules found with current thresholds.")
    else:
        rules.sort_values(["confidence", "lift"], ascending=False, inplace=True)
        rules.to_csv(OUT_DIR / "association_rules.csv", index=False)
        print(" Saved association_rules.csv")

        print("\nTop association rules:")
        print(
            rules[["antecedents", "consequents", "support", "confidence", "lift"]]
            .head()
        )

    return rules



def recommend_products(product_name: str,
                       rules_df: pd.DataFrame,
                       top_n: int = 5) -> pd.DataFrame:
    """Return top-N recommended products for a given product_name."""
    product_name = product_name.lower().strip()
    print(f"\n Generating recommendations for: '{product_name}'")

    if rules_df.empty:
        print("No rules available. Try lowering min_support or confidence.")
        return rules_df

    mask = rules_df["antecedents"].apply(lambda items: product_name in list(items))
    subset = rules_df[mask]

    if subset.empty:
        print("No rules found for this product. Try another one.")
        return subset

    subset = subset.sort_values("confidence", ascending=False)

    result = subset[["antecedents", "consequents", "support", "confidence", "lift"]].head(top_n)
    print("\nRecommended combinations:")
    print(result)

    return result


# 4. RFM SEGMENTATION

def build_rfm_table(df: pd.DataFrame) -> pd.DataFrame:
   
    print("\n Building RFM table...")

    rfm_df = df.copy()

    # Take only valid customers
    rfm_df = rfm_df.dropna(subset=["CustomerID"])

    # Reference date: one day after the last transaction
    reference_date = rfm_df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = rfm_df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Quantity", "sum")
    )

    # Clip weird negatives
    rfm["Monetary"] = rfm["Monetary"].clip(lower=0)

    # Score each dimension from 1–4 using quartiles
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1])
    rfm["F_Score"] = pd.qcut(rfm["Frequency"], 4, labels=[1, 2, 3, 4])
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4])

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    rfm.to_csv(OUT_DIR / "rfm_segments.csv")
    print(" Saved rfm_segments.csv")

    print("\nRFM sample:")
    print(rfm.head())

    return rfm


def plot_rfm_distributions(rfm: pd.DataFrame) -> None:
    """Plot simple distributions for Recency and Frequency."""
    print("\n Plotting RFM distributions...")

    plt.figure(figsize=(6, 4))
    sns.histplot(rfm["Recency"], kde=True)
    plt.title("Recency Distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "recency_dist.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(rfm["Frequency"], kde=True)
    plt.title("Frequency Distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "frequency_dist.png")
    plt.close()

    print(" Saved recency_dist.png and frequency_dist.png")


# 5. MAIN EXECUTION

def main():
    # Step 1: Load and prepare data
    orders_raw = load_orders(DATA_FILE)
    orders = prepare_base_dataframe(orders_raw)

    # Step 2: Market Basket Analysis
    basket_matrix = build_basket_matrix(orders)
    rules = run_apriori_and_rules(basket_matrix, min_support=0.01)

    # Step 3: Example recommendation
    # Change "whole milk" to any item present in your dataset
    recommend_products("whole milk", rules)

    # Step 4: RFM segmentation
    rfm = build_rfm_table(orders)
    plot_rfm_distributions(rfm)

    print(f"\n✅ All done. Check outputs here:\n{OUT_DIR}")


if __name__ == "__main__":
    main()
