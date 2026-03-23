import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle

# Load dataset
df = pd.read_csv("Mobile_Price_Prediction.csv")

# Remove useless index column if it exists
removed_columns = []
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
    removed_columns.append("Unnamed: 0")

# Encode brand column into numeric columns
brand_encoded = False
if "Brand me" in df.columns:
    df = pd.get_dummies(df, columns=["Brand me"], drop_first=True)
    brand_encoded = True

# Convert required columns to numeric
numeric_cols = [
    "Ratings",
    "RAM",
    "ROM",
    "Mobile_Size",
    "Primary_Cam",
    "Selfi_Cam",
    "Battery_Power",
    "Price"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing values
df = df.dropna()

# Features and target
X = df.drop(columns=["Price"])
y = df["Price"]

# Save feature names
feature_names = X.columns.tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"R2 Score: {r2:.4f}")
print(f"R2 Percentage: {r2 * 100:.2f}%")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

metrics = {
    "r2": r2,
    "mae": mae,
    "mse": mse,
    "rmse": rmse
}

preprocessing_info = {
    "removed_columns": removed_columns,
    "brand_encoded": brand_encoded,
    "notes": [
        "Unnamed: 0 was removed because it is only an index column.",
        "Brand me was converted into numeric dummy columns using one-hot encoding.",
        "Numeric columns were converted using pd.to_numeric(errors='coerce').",
        "Rows with missing or invalid values were removed using dropna()."
    ]
}

# Save model files
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

with open("metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

with open("preprocessing_info.pkl", "wb") as f:
    pickle.dump(preprocessing_info, f)

# Save final dataset
df.to_csv("final_dataset.csv", index=False)

# Create static folder if it does not exist
os.makedirs("static", exist_ok=True)

# Plot: actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, label="Predictions")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    label="Ideal Line"
)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.legend()
plt.tight_layout()
plt.savefig("static/model_graph.png")
plt.close()