import tensorflow as tf
from .shape_utils import get_pooling_layer
from .autoencoder import build_autoencoder
from .calibration import get_threshold_from_stats

@tf.keras.utils.register_keras_serializable(package="TensorImmune")
class ImmuneModel(tf.keras.Model):
    """
    A Keras wrapper that injects Out-of-Distribution detection natively into a 
    TensorFlow/Keras model's computational graph.
    """
    def __init__(self, base_model, monitor_layer, sensitivity=0.95, bottleneck_size=None, immune_loss_weight=1.0, **kwargs):
        super(ImmuneModel, self).__init__(**kwargs)
        
        # Validate base_model
        if not hasattr(base_model, "inputs") or not base_model.inputs:
            raise ValueError("The base model must be a Keras Sequential or Functional model. Subclassed models without explicitly defined inputs are not supported because their computational graph cannot be introspected.")
            
        try:
            target_layer = base_model.get_layer(monitor_layer)
        except ValueError:
            available_layers = [l.name for l in base_model.layers]
            raise ValueError(f"Monitor layer '{monitor_layer}' not found in the base model. Available layers: {available_layers}")
            
        self.base_model_ref = base_model
        self.monitor_layer = monitor_layer
        self.sensitivity = float(sensitivity)
        self.bottleneck_size = bottleneck_size
        self.immune_loss_weight = float(immune_loss_weight)
        
        # Safely get base model outputs
        base_outputs = base_model.outputs if hasattr(base_model, 'outputs') and base_model.outputs else [base_model.output]
        self.base_output_count = len(base_outputs)
        
        # Build feature extractor capturing both task output(s) and monitor layer
        self.feature_extractor = tf.keras.Model(
            inputs=base_model.inputs,
            outputs=base_outputs + [target_layer.output],
            name="feature_extractor"
        )
        
        # Determine pool layer based on shape
        monitor_shape = target_layer.output.shape
        rank = len(monitor_shape)
        self.pool_layer = get_pooling_layer(rank)
        
        # Calculate flattened dimension
        dummy_input = tf.keras.Input(shape=monitor_shape[1:])
        pooled_dummy = self.pool_layer(dummy_input)
        flat_dim = pooled_dummy.shape[-1]
        
        # Build symbiotic autoencoder
        self.autoencoder = build_autoencoder(flat_dim, bottleneck_size)
        
        # EMA Statistics variables for threshold calibration
        self.calib_mean = self.add_weight(name='calib_mean', shape=(), initializer='zeros', trainable=False, dtype=tf.float32)
        self.calib_var = self.add_weight(name='calib_var', shape=(), initializer='zeros', trainable=False, dtype=tf.float32)
        self.calib_count = self.add_weight(name='calib_count', shape=(), initializer='zeros', trainable=False, dtype=tf.float32)
        
        # Calibrated anomaly threshold (serialized with model)
        self.anomaly_threshold = self.add_weight(name='anomaly_threshold', shape=(), initializer='zeros', trainable=False, dtype=tf.float32)

    def _update_calibration_stats(self, batch_mse):
        batch_mean = tf.reduce_mean(batch_mse)
        # Prevent variance calculation errors if batch size is 1
        batch_size = tf.cast(tf.shape(batch_mse)[0], tf.float32)
        batch_var = tf.where(batch_size > 1.0, tf.math.reduce_variance(batch_mse), 0.0)
        
        momentum = tf.constant(0.99, dtype=tf.float32)
        
        is_first = tf.equal(self.calib_count, 0.0)
        
        def first_step():
            self.calib_mean.assign(batch_mean)
            self.calib_var.assign(batch_var)
            return tf.constant(0.0)
            
        def next_steps():
            delta = batch_mean - self.calib_mean
            new_mean = self.calib_mean + (1.0 - momentum) * delta
            new_var = momentum * self.calib_var + (1.0 - momentum) * batch_var + momentum * (1.0 - momentum) * tf.square(delta)
            self.calib_mean.assign(new_mean)
            self.calib_var.assign(new_var)
            return tf.constant(0.0)
            
        tf.cond(is_first, first_step, next_steps)
        self.calib_count.assign_add(1.0)
        
        new_threshold = get_threshold_from_stats(self.calib_mean, self.calib_var, self.sensitivity)
        self.anomaly_threshold.assign(new_threshold)

    @tf.function
    def call(self, inputs, training=False, return_score=False):
        extractor_outputs = self.feature_extractor(inputs, training=training)
        
        task_output = extractor_outputs[:-1]
        if self.base_output_count == 1:
            task_output = task_output[0]
            
        monitor_acts = extractor_outputs[-1]
        
        pooled_acts = self.pool_layer(monitor_acts)
        reconstructed = self.autoencoder(pooled_acts, training=training)
        
        mse = tf.reduce_mean(tf.square(pooled_acts - reconstructed), axis=-1)
        
        if return_score:
            return task_output, mse
        else:
            return task_output, mse > self.anomaly_threshold

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        with tf.GradientTape() as tape:
            extractor_outputs = self.feature_extractor(x, training=True)
            task_output = extractor_outputs[:-1]
            if self.base_output_count == 1:
                task_output = task_output[0]
            monitor_acts = extractor_outputs[-1]
            
            pooled_acts = self.pool_layer(monitor_acts)
            reconstructed = self.autoencoder(pooled_acts, training=True)
            
            y_pred_for_task = task_output
            if isinstance(y, dict) and isinstance(task_output, (list, tuple)):
                y_pred_for_task = dict(zip(self.base_model_ref.output_names, task_output))
                
            if hasattr(self, "compute_loss"):
                loss = self.compute_loss(x=x, y=y, y_pred=y_pred_for_task, sample_weight=sample_weight)
            else:
                loss = self.compiled_loss(y, y_pred_for_task, sample_weight=sample_weight, regularization_losses=self.losses)
            
            batch_mse = tf.reduce_mean(tf.square(pooled_acts - reconstructed), axis=-1)
            ae_loss = tf.reduce_mean(batch_mse)
            
            total_loss = loss + self.immune_loss_weight * ae_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self._update_calibration_stats(batch_mse)
        
        if hasattr(self, "compute_metrics"):
            self.compute_metrics(x=x, y=y, y_pred=y_pred_for_task, sample_weight=sample_weight)
            results = {m.name: m.result() for m in self.metrics}
        else:
            self.compiled_metrics.update_state(y, y_pred_for_task, sample_weight=sample_weight)
            results = {m.name: m.result() for m in self.metrics}
            
        results["ae_loss"] = ae_loss
        results["threshold"] = self.anomaly_threshold
        return results

    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        extractor_outputs = self.feature_extractor(x, training=False)
        task_output = extractor_outputs[:-1]
        if self.base_output_count == 1:
            task_output = task_output[0]
        monitor_acts = extractor_outputs[-1]
        
        pooled_acts = self.pool_layer(monitor_acts)
        reconstructed = self.autoencoder(pooled_acts, training=False)
        
        y_pred_for_task = task_output
        if isinstance(y, dict) and isinstance(task_output, (list, tuple)):
            y_pred_for_task = dict(zip(self.base_model_ref.output_names, task_output))
            
        if hasattr(self, "compute_loss"):
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred_for_task, sample_weight=sample_weight)
        else:
            loss = self.compiled_loss(y, y_pred_for_task, sample_weight=sample_weight, regularization_losses=self.losses)
        
        batch_mse = tf.reduce_mean(tf.square(pooled_acts - reconstructed), axis=-1)
        ae_loss = tf.reduce_mean(batch_mse)
        
        if hasattr(self, "compute_metrics"):
            self.compute_metrics(x=x, y=y, y_pred=y_pred_for_task, sample_weight=sample_weight)
            results = {m.name: m.result() for m in self.metrics}
        else:
            self.compiled_metrics.update_state(y, y_pred_for_task, sample_weight=sample_weight)
            results = {m.name: m.result() for m in self.metrics}
            
        results["ae_loss"] = ae_loss
        results["threshold"] = self.anomaly_threshold
        return results

    def predict_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data
        return self(x, training=False, return_score=True)

    def predict(self, x, *args, **kwargs):
        return_score = kwargs.pop('return_score', False)
        outputs = super().predict(x, *args, **kwargs)
        
        # Keras predict unrolls outputs. The last element is always the MSE score.
        if isinstance(outputs, list) and len(outputs) > 1:
            scores = outputs[-1]
            task_preds = outputs[:-1]
            if len(task_preds) == 1:
                task_preds = task_preds[0]
        else:
            # Fallback if outputs is somehow not unrolled
            task_preds, scores = outputs[0], outputs[1]
            
        if return_score:
            return task_preds, scores
        else:
            flags = scores > self.anomaly_threshold.numpy()
            return task_preds, flags

    def get_config(self):
        config = super().get_config()
        try:
            base_model_config = tf.keras.utils.serialize_keras_object(self.base_model_ref)
        except AttributeError:
            if hasattr(tf.keras.saving, 'serialize_keras_object'):
                base_model_config = tf.keras.saving.serialize_keras_object(self.base_model_ref)
            else:
                base_model_config = self.base_model_ref.get_config()
                
        config.update({
            "base_model": base_model_config,
            "monitor_layer": self.monitor_layer,
            "sensitivity": self.sensitivity,
            "bottleneck_size": self.bottleneck_size,
            "immune_loss_weight": self.immune_loss_weight,
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        base_model_config = config.pop("base_model")
        try:
            base_model = tf.keras.utils.deserialize_keras_object(base_model_config, custom_objects=custom_objects)
        except AttributeError:
            if hasattr(tf.keras.saving, 'deserialize_keras_object'):
                base_model = tf.keras.saving.deserialize_keras_object(base_model_config, custom_objects=custom_objects)
            else:
                if "layers" in base_model_config:
                    base_model = tf.keras.Sequential.from_config(base_model_config, custom_objects=custom_objects)
                else:
                    base_model = tf.keras.Model.from_config(base_model_config, custom_objects=custom_objects)
        return cls(base_model=base_model, **config)
