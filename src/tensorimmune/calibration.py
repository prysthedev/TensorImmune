import tensorflow as tf

def get_threshold_from_stats(mean, variance, sensitivity):
    """
    Calculates the numeric anomaly threshold based on running mean and variance and sensitivity.
    
    Args:
        mean (tf.Tensor): EMA of reconstruction error.
        variance (tf.Tensor): EMA of variance of reconstruction error.
        sensitivity (float or tf.Tensor): Percentile sensitivity (0 to 1).
        
    Returns:
        tf.Tensor: The calibrated threshold.
    """
    std = tf.sqrt(tf.maximum(variance, 0.0))
    
    # Map sensitivity (percentile) to number of standard deviations for a normal distribution
    p = tf.cast(sensitivity, tf.float32)
    p = tf.clip_by_value(p, 0.0001, 0.9999)
    z = tf.math.sqrt(2.0) * tf.math.erfinv(2.0 * p - 1.0)
    
    threshold = mean + z * std
    return threshold
