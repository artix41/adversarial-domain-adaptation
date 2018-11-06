import tensorflow as tf
from utils import leaky_relu

def feature_encoder(x, scope, config):
    """Encoder of an image into a feature space 
    
    Parameters
    ----------
    x : tensor of shape = [?, 32, 32, 3]
        Normally takes a real image (except if you use cycle-consistency)
    scope : {'source', 'target'}
        Corresponds to the domain of x

    Returns
    -------
    fc1_classif : tensor of shape = [?, 10]
        Mean of the embedding space conditionned on x
        
    """

    scope = scope + "/feature_encoder"
    initializer = tf.contrib.layers.xavier_initializer()
    ch = config["channels"]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE): # not shared
        conv1 = tf.layers.conv2d(x, ch, [5, 5], strides=2, padding='SAME', 
                                 kernel_initializer=initializer, activation=leaky_relu,
                                 name="conv1")
    
        conv2 = tf.layers.conv2d(conv1, ch*2, [5, 5], strides=2, padding='SAME', 
                                 kernel_initializer=initializer, activation=leaky_relu,
                                 name="conv2")
        
        conv3 = tf.layers.conv2d(conv2, ch*4, [8, 8], strides=1, padding='VALID', 
                                 kernel_initializer=initializer, activation=leaky_relu,
                                 name="conv3")

        conv4 = tf.layers.conv2d(conv3, ch*8, [1, 1], strides=1, padding='VALID', 
                                 kernel_initializer=initializer, activation=leaky_relu,
                                 name="conv4")
        conv4 = tf.contrib.layers.flatten(conv4)

        fc1_classif = tf.layers.dense(inputs=conv4, units=10, activation=None, kernel_initializer=initializer,
                                      name="fc1_classif")
        
    return fc1_classif