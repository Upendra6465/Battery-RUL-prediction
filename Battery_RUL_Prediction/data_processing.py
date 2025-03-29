import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import seaborn as sns

# ✅ Load dataset
data = pd.read_csv("Battery_RUL.csv")

# ✅ Define Features & Target
features = [
    "Cycle_Index",
    "Discharge Time (s)",
    "Decrement 3.6-3.4V (s)",
    "Max. Voltage Dischar. (V)",
    "Min. Voltage Charg. (V)",
    "Time at 4.15V (s)",
    "Time constant current (s)",
    "Charging time (s)",
]

# ✅ Extract Only Required Columns
X = data[features]
y = data["RUL"]  # Keep RUL only for comparison, not training

# ✅ Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Save the scaler for future use
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ✅ Group data per battery
battery_groups = data.groupby("Cycle_Index")  # Assuming each battery has a unique 'Cycle_Index' reset
sequences = []
labels = []

# ✅ Process each battery separately
for _, battery_data in battery_groups:
    battery_X = battery_data[features].values
    battery_y = battery_data["RUL"].values
    
    # ✅ Create sequences (Variable length per battery)
    for i in range(len(battery_X) - 100):  # Use all available cycles per battery
        sequences.append(battery_X[i:i+100])  # 100-cycle sequences
        labels.append(battery_y[i+100-1])  # Last RUL in sequence

X_sequences = np.array(sequences)
y_sequences = np.array(labels)

# ✅ Save Processed Data
np.save("X_sequences.npy", X_sequences)
np.save("y_sequences.npy", y_sequences)

print(f"✅ Preprocessing Complete: {X_sequences.shape[0]} samples created.")
