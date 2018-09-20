import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def log(x, eps=1e-8):
    return tf.log(x + eps)
    
def normalize(x, max_value):
    """ If x takes its values between 0 and max_value, normalize it between -1 and 1"""
    return (x / float(max_value)) * 2 - 1
    
def unnormalize(x):
    return (x + 1) / 2

def leaky_relu(x, alpha=0.05):
    return tf.maximum(x, alpha*x)

def accuracy(y_true, y_predict):
    """ shape: (n_samples,) """
    
    return tf.reduce_sum(tf.cast(tf.equal(y_true,y_predict), dtype=tf.int32)) / tf.shape(y_true)[0]


def plot_images(index, X_source, X_target, X_s2s, X_t2t, X_s2t, X_t2s, X_cycle_s2s, X_cycle_t2t, Y_source_predict, Y_target_predict):
    plt.rcParams['figure.figsize'] = (20, 10)

    plt.subplot(2,4,1)
    plt.title(Y_target_predict[index])
    plt.imshow(unnormalize(X_target[index]))
    plt.axis('off')

    plt.subplot(2,4,2)
    plt.imshow(X_t2s[index])
    plt.axis('off')

    plt.subplot(2,4,3)
    plt.title("t2t direct")
    plt.imshow(X_t2t[index].reshape(32,32,3))
    plt.axis('off')
    
    plt.subplot(2,4,4)
    plt.title("t2t cycle")
    plt.imshow(X_cycle_t2t[index].reshape(32,32,3))
    plt.axis('off')
    
    plt.subplot(2,4,5)
    plt.title(Y_source_predict[index])
    plt.imshow(unnormalize(X_source[index]))
    plt.axis('off')

    plt.subplot(2,4,6)
    plt.imshow(X_s2t[index])
    plt.axis('off')

    plt.subplot(2,4,7)
    plt.title("s2s direct")
    plt.imshow(X_s2s[index])
    plt.axis('off')
    
    plt.subplot(2,4,8)
    plt.title("s2s cycle")
    plt.imshow(X_cycle_s2s[index].reshape(32,32,3))
    plt.axis('off')
