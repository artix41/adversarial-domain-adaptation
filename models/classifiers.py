import tensorflow as tf
from utils import leaky_relu

def classifier(x, scope):
    """Classifier in the embedding space
    
    Parameters
    ----------
    x : tensor of shape = [?, 1, 1, embed_dim]
        Either the input (real sample) or the generated image (fake sample)
    config: config file for the classifier

    Returns
    -------
    fc1_classif : tensor of shape = [10]
        Output of the source classifier, without any activation (softmax used after in the loss)
        
    """

    initializer = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope(scope + "/embedding_classifier", reuse=tf.AUTO_REUSE):
        x = tf.contrib.layers.flatten(x)
        fc1_classif = tf.layers.dense(inputs=x, units=10, activation=None, kernel_initializer=initializer,
                                      name="fc1_classif")
        
    return fc1_classif
