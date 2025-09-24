import tensorflow as tf
import numpy as np
import os

# --- Configuration ---
# 1. SET THE PATH TO YOUR SAVED MODEL
# If you have a .h5 file:
MODEL_PATH = 'my_neural_network.h5' 
# If you have a SavedModel directory:
# MODEL_PATH = 'my_saved_model_directory'

# 2. SET THE PATH FOR THE OUTPUT TFLITE FILE
TFLITE_MODEL_PATH = 'model_quant.tflite'

# 3. (IMPORTANT) PROVIDE A REPRESENTATIVE DATASET
# You need a small sample of your training or validation data (e.g., 100-200 samples)
# to help the converter learn the data ranges for effective quantization.
# Make sure this data is preprocessed exactly like your training data!
# For example, load your data from a .npy file or generate it.
# Replace this with your actual data loading logic.
try:
    # Example: x_train = np.load('x_train.npy')
    # For this example, we'll create random data.
    # The shape should match your model's input shape.
    representative_data = np.random.rand(100, 32, 32, 3).astype(np.float32) # e.g., for a 32x32 RGB image model
    print("Using dummy representative data. REPLACE with your actual data for best results.")
except:
    print("Could not load representative data. Please configure the script.")
    exit()
# --- End of Configuration ---


# Define the generator function for the representative dataset
def representative_dataset_gen():
    for i in range(len(representative_data)):
        yield [np.array([representative_data[i]], dtype=np.float32)]

# Load your trained Keras model
print(f"Loading model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print("Error: Model path does not exist.")
    exit()
    
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Initialize the TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# --- Apply INT8 Quantization ---
# This is the key step for microcontrollers
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
# Enforce that the converter throws an error if it can't quantize an operation
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8 # or tf.uint8

# Convert the model
print("Starting TFLite conversion and quantization...")
try:
    tflite_quant_model = converter.convert()
    print("Conversion successful!")
except Exception as e:
    print(f"Error during conversion: {e}")
    print("\nTip: This error can happen if your model contains operations not supported by TFLite's INT8 quantization. Try a simpler model architecture if possible.")
    exit()

# Save the quantized model to a file
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_quant_model)

print(f"Quantized TFLite model saved to: {TFLITE_MODEL_PATH}")
print(f"Original model size: {os.path.getsize(MODEL_PATH) / 1024:.2f} KB")
print(f"Converted TFLite model size: {len(tflite_quant_model) / 1024:.2f} KB")