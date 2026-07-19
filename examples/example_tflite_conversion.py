import tensorflow as tf
import numpy as np
from tensorimmune import ImmuneModel

# 1. Create a base model
inputs = tf.keras.Input(shape=(10,), name="input")
x = tf.keras.layers.Dense(32, activation='relu')(inputs)
x = tf.keras.layers.Dense(16, activation='relu', name='feature_bottleneck')(x)
outputs = tf.keras.layers.Dense(2, activation='softmax', name="output")(x)
base_model = tf.keras.Model(inputs, outputs)

# 2. Wrap it with TensorImmune
model = ImmuneModel(
    base_model=base_model,
    monitor_layer='feature_bottleneck',
    sensitivity=0.95
)

# 3. Train as normal
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training model...")
x_train = np.random.normal(0, 1, size=(1000, 10)).astype(np.float32)
y_train = np.random.randint(0, 2, size=(1000,)).astype(np.float32)
model.fit(x_train, y_train, epochs=5, batch_size=32)

print(f"Calibrated Anomaly Threshold: {model.anomaly_threshold.numpy():.4f}")

# 4. Export to TFLite
print("Converting to TFLite...")
# Convert using the `call` signature (return_score=False -> boolean flag)
run_model = tf.function(lambda x: model(x, return_score=False))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 10], tf.float32)
)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

tflite_path = "tensorimmune_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"Saved TFLite model to {tflite_path}")

# 5. Run inference using TFLite
print("Running TFLite inference...")
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test with in-distribution data
x_test_id = np.random.normal(0, 1, size=(1, 10)).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], x_test_id)
interpreter.invoke()
preds_id = interpreter.get_tensor(output_details[0]['index'])
flag_id = interpreter.get_tensor(output_details[1]['index'])
print(f"In-distribution: Prediction={preds_id}, AnomalyFlag={flag_id}")

# Test with OOD data
x_test_ood = np.random.normal(10, 5, size=(1, 10)).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], x_test_ood)
interpreter.invoke()
preds_ood = interpreter.get_tensor(output_details[0]['index'])
flag_ood = interpreter.get_tensor(output_details[1]['index'])
print(f"Out-of-distribution: Prediction={preds_ood}, AnomalyFlag={flag_ood}")

print("TFLite conversion and inference successful!")
