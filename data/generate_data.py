import pandas as pd
import numpy as np
import os

np.random.seed(42)

n = 2000

data = pd.DataFrame({
    "invoice_amount": np.random.randint(1000, 10000, n),
    "avg_delay_days": np.random.randint(0, 30, n),
    "num_past_invoices": np.random.randint(1, 50, n),
    "invoice_gap_days": np.random.randint(5, 60, n),
    "industry_category": np.random.randint(0, 5, n),
    "reliability_score": np.random.uniform(0, 1, n)
})

data["late_payment"] = (
    (data["avg_delay_days"] > 10) |
    (data["reliability_score"] < 0.4)
).astype(int)

os.makedirs("data", exist_ok=True)
data.to_csv("data/invoices.csv", index=False)

print("Dataset created successfully!")
