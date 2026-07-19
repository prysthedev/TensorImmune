import tensorflow as tf

def get_pooling_layer(rank):
    """
    Returns the appropriate pooling or flattening layer based on the input tensor's rank.
    
    Args:
        rank (int): The rank (number of dimensions) of the tensor.
        
    Returns:
        tf.keras.layers.Layer: The pooling or flatten layer.
    """
    if rank == 2:
        return tf.keras.layers.Flatten()
    elif rank == 3:
        return tf.keras.layers.GlobalAveragePooling1D()
    elif rank == 4:
        return tf.keras.layers.GlobalAveragePooling2D()
    elif rank == 5:
        return tf.keras.layers.GlobalAveragePooling3D()
    else:
        return tf.keras.layers.Flatten()
