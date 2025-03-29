import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# ✅ Load dataset
data = pd.read_csv("Battery_RUL.csv")

# ✅ Drop 'RUL' column for training
X = data.drop(columns=["RUL"])
y = data["RUL"]

# ✅ Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Save the scaler for later use
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ✅ Convert Data into Sequences (100 cycles per sample)
sequence_length = 100
X_sequences, y_sequences = [], []

for i in range(len(X_scaled) - sequence_length):
    X_sequences.append(X_scaled[i : i + sequence_length])
    y_sequences.append(y[i + sequence_length])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# ✅ Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

# ✅ Define Hybrid CNN Model with Attention
def build_model():
    inputs = layers.Input(shape=(100, 8))

    x = layers.Conv1D(256, kernel_size=3, activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, kernel_size=3, activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)

    # ✅ Multi-Head Attention for Feature Learning
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn])

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation="linear")(x)

    return models.Model(inputs, outputs)

# ✅ Learning Rate Scheduling
lr_schedule = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

# ✅ Training Configuration (NO EARLY STOPPING, ENSURE 100 EPOCHS)
model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss="mae", metrics=["mae"])

# ✅ Train Model (Ensure 100 Epochs)
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[lr_schedule])

# ✅ Save Model
model.save("rul_prediction_model.h5")
print("✅ Model training complete! Saved as 'rul_prediction_model.h5'")

# ✅ Save Training History to Avoid FileNotFoundError
training_history = {
    "mae": history.history["mae"],
    "val_mae": history.history["val_mae"]
}
np.save("training_history.npy", training_history)

# ✅ Plot Accuracy (Epochs vs. MAE)
plt.figure(figsize=(8, 6))
plt.plot(range(1, 101), history.history["mae"], label="Training MAE", color="blue")
plt.plot(range(1, 101), history.history["val_mae"], label="Validation MAE", color="red")
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Training vs Validation MAE Over Epochs")
plt.legend()
plt.savefig("static/accuracy_plot.png")
plt.show()
