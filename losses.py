import tensorflow as tf

from utils import log

def original_disc_loss(D_real, D_gen, scope):
    """ Warning : take logits as input"""
    with tf.name_scope(scope):
        D_real , D_gen = tf.sigmoid(D_real), tf.sigmoid(D_gen)
        return tf.reduce_mean(-log(D_real) - log(1. - D_gen))
    
def original_gen_loss(D_gen, scope):
    """ Warning : take logits as input"""
    with tf.name_scope(scope):
        D_gen = tf.sigmoid(D_gen)
        return tf.reduce_mean(-log(D_gen))
    
def wasserstein_disc_loss(D_real, D_gen, scope):
    """ Warning : take logits as input"""
    with tf.name_scope(scope):
        return tf.reduce_mean(D_gen) - tf.reduce_mean(D_real)

def wasserstein_gen_loss(D_gen, scope):
    """ Warning : take logits as input"""
    with tf.name_scope(scope):
        return -tf.reduce_mean(D_gen)
    
def lsgan_disc_loss(D_real, D_gen, scope):
    """ Warning : take logits as input"""
    with tf.name_scope(scope):
        return 1/2 * tf.reduce_mean((D_real - 1)**2) + 1/2 * tf.reduce_mean(D_gen**2)
    
def lsgan_gen_loss(D_gen, scope):
    """ Warning : take logits as input"""
    with tf.name_scope(scope):
        return 1/2 * tf.reduce_mean((D_gen - 1)**2)
    
def cycle_loss(cycle, original, scope):
    with tf.name_scope(scope):
        return tf.losses.mean_squared_error(cycle, original)
    
def R1_reg(D_real, X_real, D_gen, X_gen, scope):
    with tf.name_scope(scope):
        grad_D_real = tf.gradients(D_real, X_real)[0]
        reg_D_real = tf.norm(tf.reshape(grad_D_real, [tf.shape(D_real)[0],-1]), axis=1, keepdims=True)
        disc_regularizer = tf.reduce_mean(tf.square(reg_D_real))
            
        return disc_regularizer
        
def latent_loss(mean, scope):
    with tf.name_scope(scope):
        return 0.5 * tf.reduce_mean(tf.square(mean))
    
def reconstruction_loss(x, x_rec, scope, norm="l2"):
    with tf.name_scope(scope):
        if norm == "l1":
            return tf.losses.absolute_difference(x, x_rec)
        elif norm == "l2":
            return tf.losses.mean_squared_error(x, x_rec)
        else:
            raise ValueError("Norm of reconstruction loss not recognized")
        
def classification_loss(D_classif, labels, scope):
    with tf.name_scope(scope):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, 
                                                                      logits=D_classif))
def entropy_loss(D_classif, scope):
    with tf.name_scope(scope):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(D_classif),
                                                                      logits=D_classif))
                                                                  
def feat_loss(D_embed, DG_embed, scope):
    with tf.name_scope(scope):
        return tf.reduce_mean(tf.losses.absolute_difference(D_embed, DG_embed))

def accuracy(y_true, y_predict, scope):
    """ shape: (n_samples,) """
    with tf.name_scope(scope):
        return tf.reduce_sum(tf.cast(tf.equal(y_true,y_predict), dtype=tf.int32)) / tf.shape(y_true)[0]