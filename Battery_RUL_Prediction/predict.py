import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import os
import time

# ✅ Load the trained model
model = tf.keras.models.load_model("rul_prediction_model.h5", custom_objects={"mae": tf.keras.metrics.MeanAbsoluteError()})

# ✅ Load the saved scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ✅ Predict RUL and Generate Explanation
def predict_rul(input_file):
    data = pd.read_csv(input_file)

    if "RUL" in data.columns:
        actual_rul = data["RUL"].values[:100]  
        data = data.drop(columns=["RUL"])

    X_scaled = scaler.transform(data)
    X_input = X_scaled[:100].reshape(1, 100, 8)

    predicted_rul = model.predict(X_input)[0][0]

    # ✅ Generate Explanation Based on RUL
    if predicted_rul > 1000:
        explanation = (
            f"🔋 The battery is in **excellent condition**, with a predicted RUL of approximately {predicted_rul:.2f} cycles."
            "This indicates that the battery has undergone minimal degradation and is functioning optimally. "
            "🔹 Key Factors: Proper charging cycles, minimal exposure to extreme temperatures, and balanced discharge rates. "
            "🔹 Recommendation: Continue regular maintenance and avoid deep discharges for extended lifespan."
        )
    elif 700 <= predicted_rul <= 1000:
        explanation = (
            f"⚡ The battery is in **good condition**, with a predicted RUL of approximately {predicted_rul:.2f} cycles. "
            "The battery is still performing well but has experienced some wear over time. "
            "🔹 Key Factors: Some charge/discharge cycles may have been more intensive, causing moderate capacity fade. "
            "🔹 Recommendation: Avoid overcharging, and consider monitoring temperature conditions during charging."
        )
    elif 400 <= predicted_rul < 700:
        explanation = (
            f"⚠️ The battery is **moderately worn out**, with a predicted RUL of around {predicted_rul:.2f} cycles. "
            "This suggests that the battery has lost a significant portion of its capacity, potentially due to repeated deep discharges or extreme operating conditions. "
            "🔹 Key Factors: Increased internal resistance, possible lithium plating, and partial electrode degradation. "
            "🔹 Recommendation: Reduce high-current draws, avoid full discharges, and check for any voltage imbalances."
        )
    else:
        explanation = (
            f"❌ The battery is **near failure**, with a predicted RUL of only {predicted_rul:.2f} cycles. "
            "This means the battery has undergone severe degradation and may soon be unable to provide reliable power. "
            "🔹 Key Factors: High cycle count, extreme thermal stress, and excessive overcharging or deep discharging. "
            "🔹 Recommendation: Consider replacing the battery, as continued usage may lead to performance issues or safety concerns."
        )
    
    return predicted_rul, actual_rul, data[:100], explanation

# ✅ Generate Dynamic Voltage Plot
def plot_voltage_vs_cycle(data, filename_prefix="voltage_plot"):
    cycle_index = data["Cycle_Index"]
    max_voltage = data["Max. Voltage Dischar. (V)"]
    min_voltage = data["Min. Voltage Charg. (V)"]

    plt.figure(figsize=(10, 5))
    plt.plot(cycle_index, max_voltage, label="Max Voltage", color='red')
    plt.plot(cycle_index, min_voltage, label="Min Voltage", color='blue')
    
    plt.xlabel("Cycle Index")
    plt.ylabel("Voltage (V)")
    plt.title("Max & Min Voltage vs. Cycle Index")
    plt.legend()
    plt.grid()

    # ✅ Create a unique filename for each uploaded file
    timestamp = int(time.time())  
    filename = f"static/{filename_prefix}_{timestamp}.png"
    
    plt.savefig(filename)
    plt.close()

    return filename  # ✅ Return the file path of the generated plot
