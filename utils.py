import tensorflow as tf

def log(x, eps=1e-8):
    return tf.log(x + eps)
    
def normalize(x, max_value):
    """ If x takes its values between 0 and max_value, normalize it between -1 and 1"""
    return (x / float(max_value)) * 2 - 1

def leaky_relu(x, alpha=0.05):
    return tf.maximum(x, alpha*x)

    
