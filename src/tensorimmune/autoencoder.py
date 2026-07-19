import tensorflow as tf

def build_autoencoder(input_dim, bottleneck_size=None):
    """
    Builds a small symbiotic autoencoder to reconstruct layer activations.
    
    Args:
        input_dim (int): The dimension of the flattened activation vector.
        bottleneck_size (int, optional): The size of the bottleneck layer. Defaults to roughly input_dim // 4.
        
    Returns:
        tf.keras.Model: The autoencoder model.
    """
    if bottleneck_size is None:
        bottleneck_size = max(1, input_dim // 4)
        
    # Use standard Keras API to build the autoencoder
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(max(bottleneck_size * 2, 2), activation='relu')(inputs)
    x = tf.keras.layers.Dense(bottleneck_size, activation='relu', name='bottleneck')(x)
    x = tf.keras.layers.Dense(max(bottleneck_size * 2, 2), activation='relu')(x)
    outputs = tf.keras.layers.Dense(input_dim, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='symbiotic_autoencoder')
    return model
