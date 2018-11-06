import tensorflow as tf
from utils import leaky_relu

def unit_pixel_discriminator(x, scope, config):
    """Discriminator for the output of the generator (pixel-space)
    
    Parameters
    ----------
    x : tensor of shape = [?, 32, 32, 3]
        Either the input (real sample) or the generated image (fake sample)
    scope : {'source', 'target'}
        Choose 'source' for separating real_source from fake_source and 'target' for separating real_target from
        fake_target. Only used for the first layer.
    config: config file for the discriminator

    Returns
    -------
    fc1_sigmoid : tensor of shape = [1]
        Output of the discriminator real vs fake with a sigmoid
    fc1_logits : tensor of shape = [1]
        Output of the discriminator real vs fake without any activation function
    fc1_classif : tensor of shape = [10]
        Output of the source classifier, without any activation (softmax used after in the loss)
        
    """

    # Configure weight sharing
    
    scope1 = scope + "/discriminator" # first layer not shared
    if config["shared_weights"] in ["weak", "none"]:
        scope2 = scope + "/discriminator"
    else:
        scope2 = "discriminator" # shared weights at the 2nd layer if strong
    if config["shared_weights"] in ["weak", "strong"]:
        scope3 = "disciminator"
    else:
        scope3 = scope + "/discriminator"
    
    # Build the network
    
    ch = config["channels"]
    initializer = tf.contrib.layers.xavier_initializer()
    
    with tf.variable_scope(scope1, reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(x, ch, [5,5], strides=1, padding='SAME', kernel_initializer=initializer, 
                                 activation=leaky_relu, name="conv1")
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        
    with tf.variable_scope(scope2, reuse=tf.AUTO_REUSE):
        conv2 = tf.layers.conv2d(conv1, ch*2, [5,5], strides=1, padding='SAME', kernel_initializer=initializer, 
                                 activation=leaky_relu, name="conv2")
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        
        conv3 = tf.layers.conv2d(conv2, ch*4, [5,5], strides=1, padding='SAME', kernel_initializer=initializer, 
                                 activation=leaky_relu, name="conv3")
        conv3 = tf.layers.max_pooling2d(conv3, 2, 2)
        
    with tf.variable_scope(scope3, reuse=tf.AUTO_REUSE): 
        conv4 = tf.layers.conv2d(conv3, ch*8, [5,5], strides=1, padding='SAME', kernel_initializer=initializer, 
                                 activation=leaky_relu, name="conv4")
        conv4 = tf.layers.max_pooling2d(conv4, 2, 2)
        fc1 = tf.contrib.layers.flatten(conv4)
        fc1_logits = tf.layers.dense(inputs=fc1, units=1, activation=None, kernel_initializer=initializer,
                                     name="fc1_logits")
        fc1_sigmoid = tf.sigmoid(fc1_logits)
        fc1_classif = tf.layers.dense(inputs=fc1, units=10, activation=None, kernel_initializer=initializer,
                                      name="fc1_classif")
        
        embedding_layer = fc1
        
    return fc1_sigmoid, fc1_logits, fc1_classif, embedding_layer

def feature_discriminator(x, scope, config):
    """Discriminator on the feature space (output of the semantic encoder)
    
    Parameters
    ----------
    x : tensor of shape = [?, size_semantic_encoder]
        Either the input (real sample) or the generated image (fake sample)
    scope : {'source', 'target'}
        Choose 'source' for separating real_source from fake_source and 'target' for separating real_target from
        fake_target. Only used for the first layer.
    config: config file for the discriminator

    Returns
    -------
    fc2_sigmoid : tensor of shape = [1]
        Output of the discriminator real vs fake with a sigmoid
    fc2_logits : tensor of shape = [1]
        Output of the discriminator real vs fake without any activation function
    """
    
    initializer = tf.contrib.layers.xavier_initializer()
    
    with tf.variable_scope(scope + "/feature_discriminator", reuse=tf.AUTO_REUSE):
        fc1 = tf.layers.dense(inputs=x, units=10, activation=tf.nn.relu, kernel_initializer=initializer, name="fc1")
        fc2 = tf.layers.dense(inputs=fc1, units=10, activation=tf.nn.relu, kernel_initializer=initializer, name="fc2")

        fc3_logits = tf.layers.dense(inputs=fc2, units=1, activation=None, kernel_initializer=initializer,
                                     name="fc3_logits")
        fc3_sigmoid = tf.sigmoid(fc3_logits)
        
    return fc3_sigmoid, fc3_logits
