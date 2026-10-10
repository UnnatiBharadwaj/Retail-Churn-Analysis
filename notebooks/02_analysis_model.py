import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

cust = pd.read_csv("data/clean/customers_clean.csv", parse_dates=["signup_date"])
orders = pd.read_csv("data/clean/orders_clean.csv", parse_dates=["order_date"])

# ---- EDA ----
print("Overall churn rate:", round(cust.churned_flag.mean()*100,1), "%")
for col in ["membership_tier","preferred_channel","email_opt_in","gender"]:
    t = cust.groupby(col)["churned_flag"].agg(["mean","size"]).round(3)
    t["mean"] = (t["mean"]*100).round(1)
    print(f"\nChurn % by {col}\n", t)

print("\nTop 5 cities by churn %")
ct = cust.groupby("city")["churned_flag"].agg(["mean","size"])
print((ct[ct["size"]>=100].sort_values("mean",ascending=False).head(5)).round(3))

print("\nChurned vs active behaviour")
print(cust.groupby("churned")[["app_logins_last_90d","support_tickets","nps_score"]].mean().round(2))

# Orders features
agg = orders.groupby("customer_id").agg(
    order_count=("order_id","count"),
    total_spend=("net_amount_inr","sum"),
    avg_order_value=("net_amount_inr","mean"),
    returns=("order_status", lambda s: (s=="Returned").sum()),
    cancels=("order_status", lambda s: (s=="Cancelled").sum()),
).reset_index()
df = cust.merge(agg, on="customer_id", how="left").fillna({"order_count":0,"total_spend":0,"avg_order_value":0,"returns":0,"cancels":0})
print("\nReturn rate by churn status (returns / orders)")
g = df.groupby("churned")[["returns","order_count"]].sum()
print((g["returns"]/g["order_count"]*100).round(1))

# ---- Charts ----
sns.set_theme(style="whitegrid")
fig,ax=plt.subplots(figsize=(7,4))
t=cust.groupby("membership_tier")["churned_flag"].mean().mul(100).reindex(["Basic","Silver","Gold"])
sns.barplot(x=t.index,y=t.values,ax=ax,color="#1F4E78")
ax.set_title("Churn rate by membership tier"); ax.set_ylabel("Churn rate (%)"); ax.set_xlabel("")
for i,v in enumerate(t.values): ax.text(i,v+0.5,f"{v:.1f}%",ha="center")
plt.tight_layout(); plt.savefig("images/churn_by_tier.png",dpi=120); plt.close()

fig,ax=plt.subplots(figsize=(7,4))
t=cust.groupby("preferred_channel")["churned_flag"].mean().mul(100).sort_values()
sns.barplot(x=t.values,y=t.index,ax=ax,color="#1F4E78")
ax.set_title("Churn rate by preferred channel"); ax.set_xlabel("Churn rate (%)"); ax.set_ylabel("")
plt.tight_layout(); plt.savefig("images/churn_by_channel.png",dpi=120); plt.close()

fig,ax=plt.subplots(figsize=(7,4))
sns.boxplot(x="churned",y="app_logins_last_90d",data=cust,ax=ax,palette=["#9DC3E6","#C00000"])
ax.set_title("App logins in last 90 days: active vs churned"); ax.set_xlabel("Churned (0 = active, 1 = churned)")
plt.tight_layout(); plt.savefig("images/logins_by_churn.png",dpi=120); plt.close()

# ---- Model (no leakage: order aggregates are from the order history only) ----
features_num = ["age","app_logins_last_90d","support_tickets","nps_score","order_count","total_spend","avg_order_value","returns","cancels"]
features_cat = ["membership_tier","preferred_channel","email_opt_in","gender"]
X = pd.get_dummies(df[features_num+features_cat], columns=features_cat, drop_first=True).astype(float)
y = df["churned_flag"]
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

models = {
 "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)),
 "Random Forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)),
}
results={}
for name,m in models.items():
    m.fit(Xtr,ytr); p=m.predict(Xte); pr=m.predict_proba(Xte)[:,1]
    results[name]=dict(accuracy=accuracy_score(yte,p),precision=precision_score(yte,p),
        recall=recall_score(yte,p),f1=f1_score(yte,p),roc_auc=roc_auc_score(yte,pr))
    print(f"\n{name}:", {k:round(v,3) for k,v in results[name].items()})
    print("Confusion matrix [[TN FP],[FN TP]]:\n", confusion_matrix(yte,p))

rf = models["Random Forest"]
imp = pd.Series(rf.named_steps["randomforestclassifier"].feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
print("\nTop 5 features (Random Forest):\n", imp.round(3))

pd.DataFrame(results).T.round(3).to_csv("data/clean/model_results.csv")
