import tensorflow as tf
from utils import leaky_relu

def unit_decoder(x, scope, config):
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
    
    scope3 = scope + "/decoder" # first layer not shared
    if config["shared_weights"] in ["weak", "none"]:
        scope2 = scope + "/decoder"
    else:
        scope2 = "decoder" # shared weights at the 2nd layer if strong
    if config["shared_weights"] in ["weak", "strong"]:
        scope1 = "decoder"
    else:
        scope1 = scope + "/decoder"
    
    # Build the network
    
    ch = config["channels"]
    initializer = tf.contrib.layers.xavier_initializer()
    
    with tf.variable_scope(scope1, reuse=tf.AUTO_REUSE): # shared weights
        deconv1 = tf.layers.conv2d_transpose(x, ch*8, [4, 4], strides=2, padding='VALID', 
                                             kernel_initializer=initializer, activation=leaky_relu,
                                             name="deconv1")

        deconv2 = tf.layers.conv2d_transpose(deconv1, ch*4, [4, 4], strides=2, padding='SAME', 
                                             kernel_initializer=initializer, activation=leaky_relu,
                                             name="deconv2")
        
    with tf.variable_scope(scope2, reuse=tf.AUTO_REUSE):
        deconv3 = tf.layers.conv2d_transpose(deconv2, ch*2, [4, 4], strides=2, padding='SAME', 
                                             kernel_initializer=initializer, activation=leaky_relu,
                                             name="deconv3")
        
    with tf.variable_scope(scope3, reuse=tf.AUTO_REUSE):
        deconv4 = tf.layers.conv2d_transpose(deconv3, ch, [4, 4], strides=2, padding='SAME', 
                                             kernel_initializer=initializer, activation=leaky_relu,
                                             name="deconv4")
        

        deconv5 = tf.layers.conv2d_transpose(deconv4, 3, [1, 1], strides=1, padding='SAME', 
                                             kernel_initializer=initializer, activation=tf.nn.tanh,
                                             name="deconv5")
    
    return deconv5
