import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("real_dr_dataset.csv")

# Rename for consistency
df.rename(columns={
    "Blood_Sugar": "Blood_Sugar",
    "Blood_Pressure": "Blood_Pressure",
    "Vision_Test_Result": "Vision_Score",
    "Age": "Age",
    "label": "Label"
}, inplace=True)

# Select relevant features
features = ["Blood_Sugar", "Blood_Pressure", "Vision_Score", "Age"]
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Save final processed data
df.to_csv("train_quantum_ready.csv", index=False)
print("✅ Saved cleaned dataset as 'train_quantum_ready.csv'")
