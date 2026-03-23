import pandas as pd
import pickle

# Load saved files
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

def get_float_input(prompt):
    while True:
        value = input(prompt).strip()
        try:
            return float(value)
        except ValueError:
            print("Error: please enter a valid number.")

print("Enter phone characteristics:")

ratings = get_float_input("Ratings: ")
ram = get_float_input("RAM: ")
rom = get_float_input("ROM: ")
mobile_size = get_float_input("Mobile_Size: ")
primary_cam = get_float_input("Primary_Cam: ")
selfi_cam = get_float_input("Selfi_Cam: ")
battery_power = get_float_input("Battery_Power: ")

user_data = {
    "Ratings": ratings,
    "RAM": ram,
    "ROM": rom,
    "Mobile_Size": mobile_size,
    "Primary_Cam": primary_cam,
    "Selfi_Cam": selfi_cam,
    "Battery_Power": battery_power
}

# Add missing dummy columns with 0
for feature in feature_names:
    if feature not in user_data:
        user_data[feature] = 0

# Create dataframe in correct column order
input_df = pd.DataFrame([user_data])[feature_names]

# Scale and predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)

print(f"\nPredicted Price: {prediction[0]:.2f}")