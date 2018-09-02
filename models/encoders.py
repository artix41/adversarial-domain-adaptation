import tensorflow as tf
from utils import leaky_relu

def unit_encoder(x, scope, config):
    """Encoder for the two GANs
    
    Parameters
    ----------
    x : tensor of shape = [?, 32, 32, 1]
        Normally takes a real image (except if you use cycle-consistency)
    scope : {'source', 'target'}
        Corresponds to the domain of x

    Returns
    -------
    mu : tensor of shape = [?, 8, 8, 1024]
        Mean of the embedding space conditionned on x
    log_sigma_sq : tensor of shape = [?, 8, 8, 1024]
        log of the variance of the embedding space conditionned on x
    z : tensor of shape = [?, 8, 8, 1024]
        Random sample generated from mu(x) and sigma(x)
        
    """
    
    # Configure weight sharing
    
    scope1 = scope + "/encoder" # first layer not shared
    if config["shared_weights"] is in ["weak", "none"]:
        scope2 = scope + "/encoder"
    else:
        scope2 = "encoder" # shared weights at the 2nd layer if strong
    if config["shared_weights"] is in ["weak", "strong"]:
        scope3 = "encoder"
    else:
        scope3 = scope + "/encoder"
    
    # Build the network
    
    ch = config["channels"]
    initializer = tf.contrib.layers.xavier_initializer()
    
    with tf.variable_scope(scope1, reuse=tf.AUTO_REUSE): # not shared
        conv1 = tf.layers.conv2d(x, ch, [5, 5], strides=2, padding='SAME', 
                                 kernel_initializer=initializer, activation=leaky_relu)
    
    with tf.variable_scope(scope2, reuse=tf.AUTO_REUSE):
        # Layer 2: 16x16x64 --> 8x8x128
        conv2 = tf.layers.conv2d(conv1, ch*2, [5, 5], strides=2, padding='SAME', 
                                 kernel_initializer=initializer, activation=leaky_relu)
        
        conv3 = tf.layers.conv2d(conv2, ch*4, [8, 8], strides=1, padding='VALID', 
                                 kernel_initializer=initializer, activation=leaky_relu)

    with tf.variable_scope(scope3, reuse=tf.AUTO_REUSE):
        conv4 = tf.layers.conv2d(conv3, ch*8 [1, 1], strides=1, padding='VALID', 
                                 kernel_initializer=initializer, activation=leaky_relu)

        mu = tf.layers.conv2d(conv4, ch*8, [1, 1], strides=1, padding='SAME', 
                              kernel_initializer=initializer, activation=None)
        
        z = mu + tf.random_normal([tf.shape(x)[0],1,1,ch*8],0,1,dtype=tf.float32) # latent space

    return mu, z
