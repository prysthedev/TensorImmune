# TensorImmune

**TensorImmune** is a lightweight Keras wrapper that injects Out-of-Distribution (OOD) detection natively into a TensorFlow/Keras model's computational graph. A single exported artifact can flag anomalous inputs at inference time without any external monitoring infrastructure!

## Overview

Deploying machine learning models to the real world involves dealing with unexpected or out-of-distribution data. Normally, OOD detection is handled by external monitoring infrastructure or complex multi-model setups. TensorImmune simplifies this by weaving a small, symbiotic autoencoder directly into your model's computational graph. 

By observing an intermediate layer's activations (the "monitor layer"), TensorImmune simultaneously trains your primary task and the autoencoder. The model calibrates an anomaly threshold during training. When deployed, the model returns both the primary prediction and an immunity score (or anomaly flag), even when converted to edge formats like TensorFlow Lite!

## Installation

Install via pip:

```bash
pip install tensorimmune
```

## Quickstart

```python
import tensorflow as tf
from tensorimmune import ImmuneModel

# 1. Define your base Keras model (Sequential or Functional)
base_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu', name='feature_bottleneck'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 2. Wrap it with ImmuneModel
# We monitor the 'feature_bottleneck' layer. 
# sensitivity=0.95 means we set the anomaly threshold to the 95th percentile 
# of the reconstruction errors seen during training.
model = ImmuneModel(
    base_model=base_model,
    monitor_layer='feature_bottleneck',
    sensitivity=0.95 
)

# 3. Compile and train normally!
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# 4. Predict returns task predictions AND anomaly flags
predictions, is_anomaly = model.predict(x_test)

# You can also get the raw continuous immunity score
predictions, immunity_scores = model.predict(x_test, return_score=True)
```

## How Symbiotic Training Works

Under the hood, `ImmuneModel` creates a `feature_extractor` that splits the graph at the `monitor_layer`. It then dynamically builds a small autoencoder to reconstruct the activations of that layer. 

By subclassing `tf.keras.Model` and overriding `train_step`, TensorImmune trains both the original task loss and the autoencoder's mean squared error (MSE) reconstruction loss simultaneously in a single backward pass. The losses are combined via the `immune_loss_weight` parameter (which defaults to `1.0`).

If the monitor layer outputs a spatial tensor (e.g. from a Conv2D layer), TensorImmune automatically inserts a GlobalAveragePooling layer before the autoencoder to flatten the representations intelligently.

## Immunity Score & Sensitivity Calibration

During training, TensorImmune maintains a running record of reconstruction error statistics (mean, variance, count) inside the model's graph. 

The `sensitivity` parameter (between 0 and 1) dictates how strictly the model should flag anomalies. It is treated as a percentile of the training distribution. Using the running statistics and the inverse error function, TensorImmune continuously calibrates a numeric anomaly threshold (e.g. `sensitivity=0.95` maps to ~1.64 standard deviations above the mean). 

At the end of training, this threshold is automatically stored as a non-trainable `tf.Variable` directly inside the model. When you save and export the model, the threshold is serialized with it!

## Edge Deployment with TFLite

Because TensorImmune builds its architecture and threshold logic using pure `tf.keras.layers` and `tf` operations, the resulting model can be converted seamlessly to TensorFlow Lite for edge devices.

```python
import tensorflow as tf

# Get concrete function for TFLite
run_model = tf.function(lambda x: model(x, return_score=False))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 28, 28, 1], tf.float32)
)

# Convert! No custom ops required.
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

with open("immune_model.tflite", "wb") as f:
    f.write(tflite_model)
```

The exported TFLite model natively returns the boolean anomaly flag as its second output, making edge integration trivial.

## Limitations

It is important to state a known limitation: **autoencoder reconstruction error on intermediate features is generally a weaker OOD signal than more rigorous methods** (like Mahalanobis distance, Energy-based scoring, or evidential deep learning). 

TensorImmune is intended as a **lightweight, drop-in first line of defense** for edge deployment where you cannot afford external monitoring infrastructure or complex multi-model pipelines. It should not be used as a replacement for rigorous safety-critical OOD research techniques.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have ideas for improvements, bug fixes, or new features.

When contributing:
1. Ensure your code passes all existing tests.
2. Add tests for new features.
3. Update the documentation as necessary.

License: MIT
