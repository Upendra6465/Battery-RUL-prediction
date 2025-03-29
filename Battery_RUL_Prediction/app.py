from flask import Flask, render_template, request
import pandas as pd
import os
from predict import predict_rul, plot_voltage_vs_cycle

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = "uploads/" + file.filename
            file.save(file_path)

            # ✅ Predict RUL and get battery data
            predicted_rul, actual_rul, battery_data, explanation = predict_rul(file_path)

            # ✅ Read CSV to extract battery-specific values
            df = pd.read_csv(file_path)

            # ✅ Extract dynamic battery values
            max_voltage = df["Max. Voltage Dischar. (V)"].max()
            min_voltage = df["Min. Voltage Charg. (V)"].min()

            # ✅ Determine battery health status dynamically
            if predicted_rul > 900:
                health_status = "Good"
            elif 600 < predicted_rul <= 900:
                health_status = "Moderate"
            else:
                health_status = "Needs Replacement"

            # ✅ Dynamic Battery Info Dictionary
            battery_info = {
                "Battery Type": "Lithium-Ion",
                "Max Voltage": f"{max_voltage:.2f}V",
                "Min Voltage": f"{min_voltage:.2f}V",
                "Health Status": health_status
            }

            # ✅ Generate performance plot dynamically based on the uploaded battery data
            voltage_plot_path = plot_voltage_vs_cycle(df)

            return render_template(
                "result.html", 
                predicted_rul=predicted_rul, 
                battery_info=battery_info, 
                explanation=explanation,
                voltage_plot=voltage_plot_path  # ✅ Pass dynamic plot path to frontend
            )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
