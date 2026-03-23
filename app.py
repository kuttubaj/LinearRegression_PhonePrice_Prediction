from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model files
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open("metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

with open("preprocessing_info.pkl", "rb") as f:
    preprocessing_info = pickle.load(f)

# Load final dataset used for training
df = pd.read_csv("final_dataset.csv")

sample_data = df.head(10).to_dict(orient="records")
columns = df.columns.tolist()

used_columns = [
    "Ratings",
    "RAM",
    "ROM",
    "Mobile_Size",
    "Primary_Cam",
    "Selfi_Cam",
    "Battery_Power",
    "Price"
]

used_sample_data = df[used_columns].head(10).to_dict(orient="records")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            ratings = float(request.form["Ratings"])
            ram = float(request.form["RAM"])
            rom = float(request.form["ROM"])
            mobile_size = float(request.form["Mobile_Size"])
            primary_cam = float(request.form["Primary_Cam"])
            selfi_cam = float(request.form["Selfi_Cam"])
            battery_power = float(request.form["Battery_Power"])

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

            input_df = pd.DataFrame([user_data])[feature_names]
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]

        except ValueError:
            error = "Invalid input. Please enter only numbers."
        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        metrics=metrics,
        sample_data=sample_data,
        columns=columns,
        preprocessing_info=preprocessing_info,
        used_columns = used_columns,
        used_sample_data = used_sample_data
    )

if __name__ == "__main__":
    app.run(debug=True)