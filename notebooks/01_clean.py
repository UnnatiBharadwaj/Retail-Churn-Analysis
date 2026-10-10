import pandas as pd, numpy as np

RAW = "data/raw/retail_customer_churn_raw_dataset.xlsx"
cust = pd.read_excel(RAW, sheet_name="Customers")
orders = pd.read_excel(RAW, sheet_name="Orders")

print("Raw customers rows:", len(cust))
print("Raw orders rows:", len(orders))

# Customers
before = len(cust)
cust = cust.drop_duplicates(subset="customer_id", keep="first")
print("Duplicate customer rows removed:", before - len(cust))

bad_age = ~cust["age"].between(18, 70)
print("Ages outside 18-70 or missing (set to median):", int(bad_age.sum()))
cust.loc[bad_age, "age"] = np.nan
cust["age"] = cust["age"].fillna(cust["age"].median()).astype(int)

gmap = {"m": "Male", "male": "Male", "f": "Female", "female": "Female", "other": "Other"}
cust["gender"] = cust["gender"].str.strip().str.lower().map(gmap).fillna("Other")

print("Missing city filled with Unknown:", int(cust["city"].isna().sum()))
cust["city"] = cust["city"].fillna("Unknown").str.strip().str.title()

print("Missing NPS left as NaN:", int(cust["nps_score"].isna().sum()))
cust["churned_flag"] = (cust["churned"] == "Yes").astype(int)

# Orders
ob = len(orders)
neg = orders["unit_price_inr"] < 0
zero = orders["quantity"] <= 0
print("Orders with negative price:", int(neg.sum()))
print("Orders with zero/invalid quantity:", int(zero.sum()))
orders = orders[~(neg | zero)].copy()
print("Orders dropped (invalid price or quantity):", ob - len(orders))

orders["discount_pct"] = orders["discount_pct"].fillna(0)
orders["payment_method"] = orders["payment_method"].fillna("Unknown")
orders["order_date"] = pd.to_datetime(orders["order_date"])
orders["net_amount_inr"] = (orders["unit_price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
orders = orders[orders["customer_id"].isin(cust["customer_id"])]
cust["signup_date"] = pd.to_datetime(cust["signup_date"])

cust.to_csv("data/clean/customers_clean.csv", index=False)
orders.to_csv("data/clean/orders_clean.csv", index=False)
print("Clean customers:", len(cust), "| Clean orders:", len(orders))
