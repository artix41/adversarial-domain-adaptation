import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from utils import unnormalize, plot_images

import losses
from losses import *
from models import encoders
from models import decoders
from models import discriminators

from utils import log
from tqdm import tqdm

class ADA:
    def __init__(self, config, output_dir, sess, verbose=2):
        self.config = config
        self.output_dir = output_dir
        self.sess = sess
        self.iter = 0
        self.verbose = verbose
        
    def build_model(self, summary_dir):
        config_gen = self.config["networks"]["generator"]
        config_disc = self.config["networks"]["discriminator"]
        config_train = self.config["train"]

        # ===================== Placeholders =====================
        
        self.ipt_source = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='ipt_source')
        self.ipt_target = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='ipt_target')
        self.labels_source = tf.placeholder(tf.int32, shape=[None, 10], name="labels_source")
        self.labels_target = tf.placeholder(tf.int32, shape=[None, 10], name="labels_target") # for the test
        self.keep_prob = tf.placeholder_with_default(1., shape=[], name="keep_prob")
        self.is_train = tf.placeholder_with_default(False, shape=[], name="is_train")
        
        # ===================== Outputs of the NN =====================
        
        encoder = getattr(encoders, self.config["networks"]["generator"]["encoder_name"])
        decoder = getattr(decoders, self.config["networks"]["generator"]["decoder_name"])
        discriminator = getattr(discriminators, self.config["networks"]["discriminator"]["name"])
        
        # Encoders
        E_mean_source, E_source = encoder(self.ipt_source, "source", config_gen)
        E_mean_target, E_target = encoder(self.ipt_target, "target", config_gen)
        
        # Decoder
        self.G_t2s = decoder(E_target, "source", config_gen) # target to source (t2s)
        self.G_s2t = decoder(E_source, "target", config_gen) # source to target (s2t)
        self.G_t2s_mean = decoder(E_mean_target, "source", config_gen)
        self.G_s2t_mean = decoder(E_mean_source, "target", config_gen)

        # VAE
        self.G_t2t = decoder(E_target, "target", config_gen) # target to target (t2t)
        self.G_s2s = decoder(E_source, "source", config_gen) # source to source (s2s)
        self.G_t2t_mean = decoder(E_mean_target, "target", config_gen)
        self.G_s2s_mean = decoder(E_mean_source, "source", config_gen)

        # Cycle
        self.G_cycle_s2s = decoder(encoder(self.G_s2t, "target", config_gen)[1], "source", config_gen)
        self.G_cycle_t2t = decoder(encoder(self.G_t2s, "source", config_gen)[1], "target", config_gen)
        self.G_cycle_s2s_mean = decoder(encoder(self.G_s2t_mean, "target", config_gen)[0], "source", config_gen)
        self.G_cycle_t2t_mean = decoder(encoder(self.G_t2s_mean, "source", config_gen)[0], "target", config_gen)
        
        # Discriminator
        D_target, D_target_logits, D_target_classif, D_target_embed = discriminator(self.ipt_target, "target", config_disc)
        D_source, D_source_logits, D_source_classif, D_source_embed = discriminator(self.ipt_source, "source", config_disc)

        DG_t2s, DG_t2s_logits, DG_t2s_classif, DG_t2s_embed = discriminator(self.G_t2s, "source", config_disc)
        DG_s2t, DG_s2t_logits, DG_s2t_classif, DG_s2t_embed = discriminator(self.G_s2t, "target", config_disc)
        DG_s2s, DG_s2s_logits, DG_s2s_classif, DG_s2s_embed = discriminator(self.G_s2s, "source", config_disc)
        DG_t2t, DG_t2t_logits, DG_t2t_classif, DG_t2t_embed = discriminator(self.G_t2t, "target", config_disc)

        DG_t2s_mean, DG_t2s_logits_mean, DG_t2s_classif_mean, DG_t2s_embed_mean = discriminator(self.G_t2s_mean, "source", config_disc)
        DG_s2t_mean, DG_s2t_logits_mean, DG_s2t_classif_mean, DG_s2t_embed_mean = discriminator(self.G_s2t_mean, "target", config_disc)
        DG_s2s_mean, DG_s2s_logits_mean, DG_s2s_classif_mean, DG_s2s_embed_mean = discriminator(self.G_s2s_mean, "source", config_disc)
        DG_t2t_mean, DG_t2t_logits_mean, DG_t2t_classif_mean, DG_t2t_embed_mean = discriminator(self.G_t2t_mean, "target", config_disc)
        
        self.DG_t2s_predict = tf.argmax(tf.nn.softmax(DG_t2s_classif), axis=1)
        # self.DG_t2s_predict_mean = tf.argmax(tf.nn.softmax(DG_t2s_classif_mean), axis=1)
        self.D_source_predict = tf.argmax(tf.nn.softmax(D_source_classif), axis=1)
        
        # ===================== Losses =====================
        
        vae_rec_s2s_loss = reconstruction_loss(self.ipt_source, self.G_s2s, scope="Rec_s2s_loss")
        vae_kl_s2s_loss = latent_loss(E_mean_source, scope="KL_s2s_loss")

        vae_rec_t2t_loss = reconstruction_loss(self.ipt_target, self.G_t2t, scope="VAE_rec_t2t_loss")
        vae_kl_t2t_loss = latent_loss(E_mean_target, scope="KL_t2t_loss")
    
        classif_source_loss = classification_loss(D_source_classif, self.labels_source, scope="Classif_source_loss")
        classif_vae_loss = classification_loss(DG_s2s_classif, self.labels_source, scope="Classif_VAE_loss")
        
        feat_t2s_loss = feat_loss(D_source_embed, DG_t2s_embed, scope="Feature_t2s_loss")
        feat_s2t_loss = feat_loss(D_target_embed, DG_s2t_embed, scope="Feature_s2t_loss")
        
        discreg_s2s_loss = R1_reg(D_source_logits, self.ipt_source, DG_s2s_logits, self.G_s2s, scope="R1_reg_s2s_loss")
        discreg_s2t_loss = R1_reg(D_target_logits, self.ipt_target, DG_s2t_logits, self.G_s2t, scope="R1_reg_s2t_loss")
        discreg_t2s_loss = R1_reg(D_source_logits, self.ipt_source, DG_t2s_logits, self.G_t2s, scope="R1_reg_t2s_loss")
        discreg_t2t_loss = R1_reg(D_target_logits, self.ipt_target, DG_t2t_logits, self.G_t2t, scope="R1_reg_t2t_loss")
        
        cycle_s2s_loss = cycle_loss(self.G_cycle_s2s_mean, self.ipt_source, scope="Cycle_s2s_loss")
        cycle_t2t_loss = cycle_loss(self.G_cycle_t2t_mean, self.ipt_target, scope="Cycle_t2t_loss")
        
        entropy_t2s_loss = entropy_loss(DG_t2s_classif, scope="Entropy_t2s_loss")
        entropy_s2s_loss = entropy_loss(DG_s2s_classif, scope="Entropy_s2s_loss")
        self.entropy_t2s_loss = entropy_t2s_loss
        
        gen_loss_func = getattr(losses, config_train["gan_loss"] + "_gen_loss")
        disc_loss_func = getattr(losses, config_train["gan_loss"] + "_disc_loss")
        
        D_s2t_loss = disc_loss_func(D_target_logits, DG_s2t_logits, scope="GAN_discriminator_s2t_loss") 
        D_t2t_loss = disc_loss_func(D_target_logits, DG_t2t_logits, scope="GAN_discriminator_t2t_loss")

        G_s2t_loss = gen_loss_func(DG_s2t_logits, scope="GAN_generator_s2t_loss")
        G_t2t_loss = gen_loss_func(DG_t2t_logits, scope="GAN_generator_t2t_loss")

        D_t2s_loss = disc_loss_func(D_source_logits, DG_t2s_logits, scope="GAN_discriminator_t2s_loss")
        D_s2s_loss = disc_loss_func(D_source_logits, DG_s2s_logits, scope="GAN_discriminator_s2s_loss")

        G_t2s_loss = gen_loss_func(DG_t2s_logits, scope="GAN_generator_t2s_loss") 
        G_s2s_loss = gen_loss_func(DG_s2s_logits, scope="GAN_generator_s2s_loss")
        
        self.accuracy = accuracy(tf.argmax(self.labels_target, axis=1), self.DG_t2s_predict, scope="Accuracy")
        self.accuracy_mean = accuracy(tf.argmax(self.labels_target, axis=1), self.DG_t2s_predict_mean)
        
        self.D_loss = config_train["loss_weight"]["disc"] * (D_s2s_loss + D_t2t_loss + D_s2t_loss + D_t2s_loss) \
                         + config_train["loss_weight"]["feat_matching"] * (feat_t2s_loss + feat_s2t_loss) \
                         + config_train["loss_weight"]["classif_source"] * classif_source_loss \
                         + config_train["loss_weight"]["r1_reg"] * (discreg_s2s_loss + discreg_t2t_loss + discreg_s2t_loss + discreg_t2s_loss)
        
        self.G_loss = config_train["loss_weight"]["gen"] * (G_s2s_loss + G_t2t_loss + G_s2t_loss + G_t2s_loss) \
                        + config_train["loss_weight"]["vae_rec"] * (vae_rec_s2s_loss + vae_rec_t2t_loss) \
                        + config_train["loss_weight"]["vae_kl"] * (vae_kl_s2s_loss + vae_kl_t2t_loss) \
                        + config_train["loss_weight"]["classif_vae"] * classif_vae_loss \
                        + config_train["loss_weight"]["entropy"] * (entropy_t2s_loss + entropy_s2s_loss) \
                        + config_train["loss_weight"]["cycle"] * (cycle_s2s_loss + cycle_t2t_loss)
                      
        # ===================== Summary variables =====================
                      
        tf.summary.scalar("entropy_t2s_loss", entropy_t2s_loss)
        tf.summary.scalar("entropy_s2s_loss", entropy_s2s_loss)
        tf.summary.scalar("vae_rec_s2s_loss", vae_rec_s2s_loss)
        tf.summary.scalar("vae_kl_s2s_loss", vae_kl_s2s_loss)
        tf.summary.scalar("vae_rec_t2t_loss", vae_rec_t2t_loss)
        tf.summary.scalar("vae_kl_t2t_loss", vae_kl_t2t_loss)
        tf.summary.scalar("classif_source_loss", classif_source_loss)
        tf.summary.scalar("classif_vae_loss", classif_vae_loss)
        tf.summary.scalar("feat_t2s_loss", feat_t2s_loss)
        tf.summary.scalar("feat_s2t_loss", feat_s2t_loss)
        tf.summary.scalar("discreg_s2s_loss", discreg_s2s_loss)
        tf.summary.scalar("discreg_s2t_loss", discreg_s2t_loss)
        tf.summary.scalar("discreg_t2s_loss", discreg_t2s_loss)
        tf.summary.scalar("discreg_t2t_loss", discreg_t2t_loss)
        tf.summary.scalar("cycle_s2s_loss", cycle_s2s_loss)
        tf.summary.scalar("cycle_t2t_loss", cycle_t2t_loss)
        tf.summary.scalar("D_s2s_loss", D_s2s_loss)
        tf.summary.scalar("D_s2t_loss", D_s2t_loss)
        tf.summary.scalar("D_t2s_loss", D_t2s_loss)
        tf.summary.scalar("D_t2t_loss", D_t2t_loss)
        tf.summary.scalar("G_s2s_loss", G_s2s_loss)
        tf.summary.scalar("G_s2t_loss", G_s2t_loss)
        tf.summary.scalar("G_t2s_loss", G_t2s_loss)
        tf.summary.scalar("G_t2t_loss", G_t2t_loss)
        tf.summary.scalar("G_loss", self.G_loss)
        tf.summary.scalar("D_loss", self.D_loss)
        tf.summary.scalar("accuracy", self.accuracy)
        tf.summary.scalar("accuracy_mean", self.accuracy_mean)
        
        self.losses_summary = tf.summary.merge_all()
        
        D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='source/discriminator') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target/discriminator') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='source/encoder') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target/encoder') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='source/decoder') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target/decoder') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
                 
        wc = config_train["weight_clipping"]
        self.clip_D = [p.assign(tf.clip_by_value(p, -wc, wc)) for p in D_vars]

        gen = config_train["gen_opti"]
        disc = config_train["disc_opti"]
        lr_gen = config_train["lr_gen"]
        lr_disc = config_train["lr_disc"]

        self.D_solver = tf.train.AdamOptimizer(learning_rate=float(lr_disc), beta1=0.5).minimize(self.D_loss, var_list=D_vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=float(lr_gen), beta1=0.5).minimize(self.G_loss, var_list=G_vars)
        
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)

        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter(summary_dir["train"], self.sess.graph)
        self.test_writer = tf.summary.FileWriter(summary_dir["test"], self.sess.graph)
                 
    def train(self, X_source, X_target, Y_source, Y_target, summary_dir):
        config_train = self.config["train"]
        nb_iter = config_train["nb_iter"]
        nb_iter_g = config_train["nb_iter_g"]
        nb_iter_d = config_train["nb_iter_d"]
        batch_size = config_train["batch_size"]
        save_every = config_train["save_every"]
        test_every = config_train["test_every"]

        for self.iter in range(self.iter, nb_iter):
            for k in range(nb_iter_d):
                idx_sample_source = np.random.choice(len(X_source), batch_size)
                sample_source = X_source[idx_sample_source]
                sample_source_labels = Y_source[idx_sample_source]
                idx_sample_target = np.random.choice(len(X_target), batch_size)
                sample_target = X_target[idx_sample_target]
                sample_target_labels = Y_source[idx_sample_target]
                
                feed_dict = {self.ipt_source: sample_source, 
                             self.ipt_target: sample_target,
                             self.labels_source: sample_source_labels, 
                             self.labels_target: sample_target_labels,
                             self.is_train: True}

                self.sess.run(self.D_solver, feed_dict=feed_dict)
                if config_train["weight_clipping"] > 0:
                    self.sess.run(self.clip_D)
                
            for k in range(nb_iter_g):
                idx_sample_source = np.random.choice(len(X_source), batch_size)
                sample_source = X_source[idx_sample_source]
                sample_source_labels = Y_source[idx_sample_source]
                sample_target = X_target[np.random.choice(len(X_target), batch_size)]
                
                feed_dict = {self.ipt_source: sample_source, 
                             self.ipt_target: sample_target,
                             self.labels_source: sample_source_labels, 
                             self.is_train: True}
                             
                self.sess.run(self.G_solver, feed_dict=feed_dict)
            
            # Collect summary
            idx_sample_source = np.random.choice(len(X_source), batch_size)
            idx_sample_target = np.random.choice(len(X_target), batch_size)
            sample_source = X_source[idx_sample_source]
            sample_target = X_target[idx_sample_target]
            sample_source_labels = Y_source[idx_sample_source]
            sample_target_labels = Y_target[idx_sample_target]
            feed_dict = {self.ipt_source: sample_source, self.ipt_target: sample_target,
                         self.labels_source: sample_source_labels, self.labels_target: sample_target_labels,
                         self.is_train: False}
                         
            accuracy, entropy, summary = self.sess.run([self.accuracy, self.entropy_t2s_loss, self.losses_summary], 
                                                   feed_dict=feed_dict)
                                                   
            self.train_writer.add_summary(summary, global_step=self.iter)

            # Display
            if self.verbose >= 2:
                print("Iter: {} / {} -- Accuracy: {:05f} -- Entropy: {:05f}"
                      .format(self.iter, nb_iter, accuracy, entropy), end="\r")
                      
            if self.iter != 0 and (self.iter % test_every == 0 or self.iter % save_every == 0):
                print()
                return
        print()          
        self.iter += 1
                        
    def test(self, X_source, X_target, Y_source, Y_target, test_all=False):
        batch_size = self.config["test"]["batch_size"]
        # Collect summary
        idx_sample_source = np.random.choice(len(X_source), batch_size)
        idx_sample_target = np.random.choice(len(X_target), batch_size)
        sample_source = X_source[idx_sample_source]
        sample_target = X_target[idx_sample_target]
        sample_source_labels = Y_source[idx_sample_source]
        sample_target_labels = Y_target[idx_sample_target]
        feed_dict = {self.ipt_source: sample_source, self.ipt_target: sample_target,
                     self.labels_source: sample_source_labels, self.labels_target: sample_target_labels,
                     self.is_train: False}

        accuracy, entropy, summary = self.sess.run([self.accuracy, self.entropy_t2s_loss, self.losses_summary], 
                                               feed_dict=feed_dict)
                                               
        self.test_writer.add_summary(summary, global_step=self.iter)
        
        # Display
        if self.verbose >= 2:      
            print("Test -- Accuracy: {:05f} -- Entropy: {:05f}\n"
                  .format(accuracy, entropy))
                  
        if test_all:
            Y_target_predict = []

            for start in tqdm(range(0, X_target.shape[0], batch_size)):
                target_predict = self.DG_t2s_predict
                test_samples = X_target[start:batch_size+start]
                Y_target_predict = np.concatenate([Y_target_predict, 
                                                   self.sess.run(target_predict, feed_dict={self.ipt_target: test_samples})])
            return accuracy_score(np.argmax(Y_target, axis=1), Y_target_predict)
    
    def save_images(self, X_source, X_target, images_dir, nb_images=64):
        images_dir = os.path.join(self.output_dir, "generated-images")
        if not os.path.exists(images_dir):
            os.mkdir(images_dir)
        X_s2s = unnormalize(self.sess.run(self.G_s2s, feed_dict={self.ipt_source: X_source[:nb_images]}))
        X_t2t = unnormalize(self.sess.run(self.G_t2t, feed_dict={self.ipt_target: X_target[:nb_images]}))
        X_s2t = unnormalize(self.sess.run(self.G_s2t, feed_dict={self.ipt_source: X_source[:nb_images]}))
        X_t2s = unnormalize(self.sess.run(self.G_t2s, feed_dict={self.ipt_target: X_target[:nb_images]}))
        X_cycle_s2s = unnormalize(self.sess.run(self.G_cycle_s2s, feed_dict={self.ipt_source: X_source[:nb_images]}))
        X_cycle_t2t = unnormalize(self.sess.run(self.G_cycle_t2t, feed_dict={self.ipt_target: X_target[:nb_images]}))
        Y_source_predict = self.sess.run(self.D_source_predict, feed_dict={self.ipt_source: X_source[:nb_images]})
        Y_target_predict = self.sess.run(self.DG_t2s_predict, feed_dict={self.ipt_target: X_target[:nb_images]})
        
        for index in range(len(X_s2s)):
            plot_images(index, X_source, X_target, X_s2s, X_t2t, X_s2t, X_t2s, X_cycle_s2s, X_cycle_t2t, Y_source_predict, Y_target_predict)
            print("Saving 'generated-images/iter_{:05d}_image_{:02d}.png'".format(self.iter, index), end="\r")
            plt.savefig(os.path.join(images_dir, "iter_{:05d}_image_{:02d}.png".format(self.iter, index)))
                  
    def save_model(self, model_dir):
        self.saver.save(self.sess, os.path.join(model_dir, "model.ckpt"))
