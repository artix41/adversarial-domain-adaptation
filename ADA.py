import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from utils import unnormalize, plot_images

import losses
from losses import *
from models import feature_encoders, generators, discriminators, classifiers

from utils import log
from tqdm import tqdm

class ADA:
    tf_optimizer = {"adam": tf.train.AdamOptimizer,
                    "rmsprop": tf.train.RMSPropOptimizer}

    def __init__(self, config, output_dir, sess, verbose=2):
        self.config = config
        self.output_dir = output_dir
        self.sess = sess
        self.iter = 0
        self.verbose = verbose
        
    def build_model(self, summary_dir):
        config_gen = self.config["networks"]["generator"]
        config_disc = self.config["networks"]["pixel_discriminator"]
        config_train = self.config["train"]

        gen_optimizer = self.config["train"]["gen_opti"]
        disc_optimizer = self.config["train"]["disc_opti"]
        fs_optimizer = self.config["train"]["fs_opti"]

        # ===================== Placeholders =====================
        
        self.ipt_source = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='ipt_source')
        self.ipt_target = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='ipt_target')
        self.labels_source = tf.placeholder(tf.int32, shape=[None, 10], name="labels_source")
        self.labels_target = tf.placeholder(tf.int32, shape=[None, 10], name="labels_target") # for the test
        self.keep_prob = tf.placeholder_with_default(1., shape=[], name="keep_prob")
        self.is_train = tf.placeholder_with_default(False, shape=[], name="is_train")
        
        # ===================== Outputs of the networks =====================
        
        encoder = getattr(generators, self.config["networks"]["generator"]["encoder_name"])
        decoder = getattr(generators, self.config["networks"]["generator"]["decoder_name"])
        feature_encoder = getattr(feature_encoders, self.config["networks"]["feature_encoder"]["name"])
        pixel_discriminator = getattr(discriminators, self.config["networks"]["pixel_discriminator"]["name"])
        feature_discriminator = getattr(discriminators, self.config["networks"]["feature_discriminator"]["name"])
        latent_classifier = getattr(classifiers, self.config["networks"]["latent_classifier"]["name"])

        # Encoders
        E_mean_source, E_random_source = encoder(self.ipt_source, "source", config_gen)
        E_mean_target, E_random_target = encoder(self.ipt_target, "target", config_gen)
        
        # Decoders
        G_t2s_random = decoder(E_random_target, "source", config_gen) # target to source (t2s)
        G_s2t_random = decoder(E_random_source, "target", config_gen) # source to target (s2t)
        G_t2t_random = decoder(E_random_target, "target", config_gen) # target to target (t2t)
        G_s2s_random = decoder(E_random_source, "source", config_gen) # source to source (s2s)

        self.G_t2s_mean = decoder(E_mean_target, "source", config_gen)
        self.G_s2t_mean = decoder(E_mean_source, "target", config_gen)
        self.G_t2t_mean = decoder(E_mean_target, "target", config_gen)
        self.G_s2s_mean = decoder(E_mean_source, "source", config_gen)

        if self.config["networks"]["generator"]["random_latent_space"]:
            self.G_t2s = G_t2s_random
            self.G_s2t = G_s2t_random
            self.G_t2t = G_t2t_random
            self.G_s2s = G_s2s_random
        else:
            self.G_t2s = self.G_t2s_mean
            self.G_s2t = self.G_s2t_mean
            self.G_t2t = self.G_t2t_mean
            self.G_s2s = self.G_s2s_mean

        # Feature space encoders
        FS_source = feature_encoder(self.ipt_source, "source", self.config["networks"]["feature_encoder"]) # F_s applied to source input
        FS_target = feature_encoder(self.ipt_target, "source", self.config["networks"]["feature_encoder"]) # F_s applied to target input
        FS_s2t = feature_encoder(self.G_s2t, "source", self.config["networks"]["feature_encoder"]) # F_s applied to G_s2t
        FS_t2s = feature_encoder(self.G_t2s, "source", self.config["networks"]["feature_encoder"]) # F_s applied to G_t2s

        FT_target = feature_encoder(self.ipt_target, "target", self.config["networks"]["feature_encoder"])
        FT_s2t = feature_encoder(self.G_s2t, "target", self.config["networks"]["feature_encoder"])

        # Cycle
        self.G_cycle_s2s = decoder(encoder(self.G_s2t, "target", config_gen)[1], "source", config_gen)
        self.G_cycle_t2t = decoder(encoder(self.G_t2s, "source", config_gen)[1], "target", config_gen)
        
        # Pixel Discriminator
        D_target, D_target_logits, D_target_classif, D_target_embed = pixel_discriminator(self.ipt_target, "target", config_disc)
        D_source, D_source_logits, D_source_classif, D_source_embed = pixel_discriminator(self.ipt_source, "source", config_disc)

        DG_t2s, DG_t2s_logits, DG_t2s_classif, DG_t2s_embed = pixel_discriminator(self.G_t2s, "source", config_disc)
        DG_s2t, DG_s2t_logits, DG_s2t_classif, DG_s2t_embed = pixel_discriminator(self.G_s2t, "target", config_disc)
        DG_s2s, DG_s2s_logits, DG_s2s_classif, DG_s2s_embed = pixel_discriminator(self.G_s2s, "source", config_disc)
        DG_t2t, DG_t2t_logits, DG_t2t_classif, DG_t2t_embed = pixel_discriminator(self.G_t2t, "target", config_disc)

        DG_t2s_mean, DG_t2s_logits_mean, DG_t2s_classif_mean, DG_t2s_embed_mean = pixel_discriminator(self.G_t2s_mean, "source", config_disc)
        DG_s2t_mean, DG_s2t_logits_mean, DG_s2t_classif_mean, DG_s2t_embed_mean = pixel_discriminator(self.G_s2t_mean, "target", config_disc)
        DG_s2s_mean, DG_s2s_logits_mean, DG_s2s_classif_mean, DG_s2s_embed_mean = pixel_discriminator(self.G_s2s_mean, "source", config_disc)
        DG_t2t_mean, DG_t2t_logits_mean, DG_t2t_classif_mean, DG_t2t_embed_mean = pixel_discriminator(self.G_t2t_mean, "target", config_disc)

        self.DG_t2s_predict = tf.argmax(tf.nn.softmax(DG_t2s_classif), axis=1)
        self.DG_t2s_predict_mean = tf.argmax(tf.nn.softmax(DG_t2s_classif_mean), axis=1)
        self.D_source_predict = tf.argmax(tf.nn.softmax(D_source_classif), axis=1)

        # Feature Discriminator

        D_FT_target, D_FT_target_logits = feature_discriminator(FT_target, "target", config_disc)
        D_FT_s2t, D_FT_s2t_logits = feature_discriminator(FT_s2t, "target", config_disc)

        # Latent classifier (latent space of the VAE)

        if self.config["networks"]["generator"]["random_latent_space"]:
            E_source_classif = latent_classifier(E_random_source, "source", self.config["networks"]["latent_classifier"])
            E_target_classif = latent_classifier(E_random_target, "target", self.config["networks"]["latent_classifier"])
        else:
            E_source_classif = latent_classifier(E_mean_source, "source", self.config["networks"]["latent_classifier"])
            E_target_classif = latent_classifier(E_mean_target, "target", self.config["networks"]["latent_classifier"])
        
        self.E_source_predict = tf.argmax(tf.nn.softmax(E_source_classif), axis=1)
        self.E_target_predict = tf.argmax(tf.nn.softmax(E_target_classif), axis=1)

        self.latent_source_accuracy = accuracy(tf.argmax(self.labels_source, axis=1), self.E_source_predict, scope="Latent_source_accuracy")
        self.latent_target_accuracy = accuracy(tf.argmax(self.labels_target, axis=1), self.E_target_predict, scope="Latent_target_accuracy")

        # Predictions with the feature encoder

        self.FS_source_predict = tf.argmax(tf.nn.softmax(FS_source), axis=1)
        self.FT_s2t_predict = tf.argmax(tf.nn.softmax(FT_s2t), axis=1)
        self.FT_target_predict = tf.argmax(tf.nn.softmax(FT_target), axis=1)

        self.FS_source_accuracy = accuracy(tf.argmax(self.labels_source, axis=1), self.FS_source_predict, scope="FS_source_accuracy")
        self.FT_s2t_accuracy = accuracy(tf.argmax(self.labels_target, axis=1), self.FT_s2t_predict, scope="Latent_target_accuracy")
        self.FT_target_accuracy = accuracy(tf.argmax(self.labels_target, axis=1), self.FT_target_predict, scope="FT_target_accuracy")

        # ===================== Losses =====================

        vae_rec_s2s_loss = reconstruction_loss(self.ipt_source, self.G_s2s, scope="Rec_s2s_loss")
        vae_kl_source_loss = latent_loss(E_mean_source, scope="KL_source_loss")

        vae_rec_t2t_loss = reconstruction_loss(self.ipt_target, self.G_t2t, scope="Rec_t2t_loss")
        vae_kl_target_loss = latent_loss(E_mean_target, scope="KL_target_loss")

        vae_rec_s2t_loss = reconstruction_loss(self.ipt_source, self.G_s2t, scope="Rec_s2t_loss")
        vae_rec_t2s_loss = reconstruction_loss(self.ipt_target, self.G_t2s, scope="Rec_t2s_loss")
    
        D_classif_source_loss = classification_loss(D_source_classif, self.labels_source, scope="D_classif_source_loss")
        D_classif_vae_loss = classification_loss(DG_s2s_classif, self.labels_source, scope="Disc_classif_VAE_loss")

        FS_source_classif_loss = classification_loss(FS_source, self.labels_source, scope="Feat_classif_source_loss")
        FT_s2t_classif_loss = classification_loss(FT_s2t, self.labels_source, scope="Feat_classif_source_loss")

        classif_embedding_loss = classification_loss(E_source_classif, self.labels_source, scope="Embedding_classif_source_loss")
        entropy_embedding_target_loss = entropy_loss(E_target_classif, scope="Entropy_embedding_target_loss")
        
        feat_matching_t2s_loss = feat_matching_loss(D_source_embed, DG_t2s_embed, scope="Feature_matching_t2s_loss")
        feat_matching_s2t_loss = feat_matching_loss(D_target_embed, DG_s2t_embed, scope="Feature_matching_s2t_loss")
        
        discreg_s2s_loss = R1_reg(D_source_logits, self.ipt_source, DG_s2s_logits, self.G_s2s, scope="R1_reg_s2s_loss")
        discreg_s2t_loss = R1_reg(D_target_logits, self.ipt_target, DG_s2t_logits, self.G_s2t, scope="R1_reg_s2t_loss")
        discreg_t2s_loss = R1_reg(D_source_logits, self.ipt_source, DG_t2s_logits, self.G_t2s, scope="R1_reg_t2s_loss")
        discreg_t2t_loss = R1_reg(D_target_logits, self.ipt_target, DG_t2t_logits, self.G_t2t, scope="R1_reg_t2t_loss")
        
        cycle_s2s_loss = cycle_loss(self.G_cycle_s2s, self.ipt_source, scope="Cycle_s2s_loss")
        cycle_t2t_loss = cycle_loss(self.G_cycle_t2t, self.ipt_target, scope="Cycle_t2t_loss")
        
        entropy_t2s_loss = entropy_loss(DG_t2s_classif, scope="Entropy_t2s_loss")
        entropy_s2s_loss = entropy_loss(DG_s2s_classif, scope="Entropy_s2s_loss")
        self.entropy_t2s_loss = entropy_t2s_loss

        semantic_s2t_loss = semantic_loss(FS_source, FS_s2t, scope="semantic_consistency_s2t_loss")
        semantic_t2s_loss = semantic_loss(FS_target, FS_t2s, scope="semantic_consistency_t2s_loss")

        gen_loss_func = getattr(losses, config_train["gan_loss"] + "_gen_loss")
        disc_loss_func = getattr(losses, config_train["gan_loss"] + "_disc_loss")
        
        D_s2t_loss = disc_loss_func(D_target_logits, DG_s2t_logits, scope="D_s2t_loss") 
        D_t2t_loss = disc_loss_func(D_target_logits, DG_t2t_logits, scope="D_t2t_loss")
        D_t2s_loss = disc_loss_func(D_source_logits, DG_t2s_logits, scope="D_t2s_loss")
        D_s2s_loss = disc_loss_func(D_source_logits, DG_s2s_logits, scope="D_s2s_loss")

        G_s2t_loss = gen_loss_func(DG_s2t_logits, scope="G_s2t_loss")
        G_t2t_loss = gen_loss_func(DG_t2t_logits, scope="G_t2t_loss")
        G_t2s_loss = gen_loss_func(DG_t2s_logits, scope="G_t2s_loss") 
        G_s2s_loss = gen_loss_func(DG_s2s_logits, scope="G_s2s_loss")

        D_FT_s2t_loss = disc_loss_func(D_FT_target_logits, D_FT_s2t_logits, scope="D_FT_s2t_loss")

        G_FT_s2t_loss = gen_loss_func(D_FT_s2t_logits, scope="G_FT_s2t_loss")
        
        self.accuracy = accuracy(tf.argmax(self.labels_target, axis=1), self.DG_t2s_predict, scope="Accuracy")
        self.accuracy_mean = accuracy(tf.argmax(self.labels_target, axis=1), self.DG_t2s_predict_mean, scope="Accuracy_mean")
        
        weight_disc = config_train["loss_weight"]["discriminator"]
        weight_gen = config_train["loss_weight"]["generator"]
        weight_init = config_train["init"]

        self.D_loss = weight_disc["s2s"]["pix_gan"] * D_s2s_loss + weight_disc["t2t"]["pix_gan"] * D_t2t_loss \
                    + weight_disc["s2t"]["pix_gan"] * D_s2t_loss + weight_disc["t2s"]["pix_gan"] * D_t2s_loss \
                    + weight_disc["s2t"]["feat_gan"] * D_FT_s2t_loss \
                    + weight_disc["s2s"]["r1_reg"] * discreg_s2s_loss + weight_disc["t2t"]["r1_reg"] * discreg_t2t_loss \
                    + weight_disc["s2t"]["r1_reg"] * discreg_s2t_loss + weight_disc["t2s"]["r1_reg"] * discreg_t2s_loss \
                    + weight_disc["s2t"]["feat_matching"] * feat_matching_s2t_loss + weight_disc["t2s"]["feat_matching"] * feat_matching_t2s_loss \
                    + weight_disc["classif_source"] * D_classif_source_loss
        
        self.G_loss = weight_gen["s2s"]["pix_gan"] * G_s2s_loss + weight_gen["t2t"]["pix_gan"] * G_t2t_loss \
                    + weight_gen["s2t"]["pix_gan"] * G_s2t_loss + weight_gen["t2s"]["pix_gan"] * G_t2s_loss \
                    + weight_gen["s2t"]["semantic_consistency"] * semantic_s2t_loss \
                    + weight_gen["t2s"]["semantic_consistency"] * semantic_t2s_loss \
                    + weight_gen["s2s"]["vae_rec"] * vae_rec_s2s_loss + weight_gen["t2t"]["vae_rec"] * vae_rec_t2t_loss \
                    + weight_gen["s2s"]["vae_kl"] * vae_kl_source_loss + weight_gen["t2t"]["vae_kl"] * vae_kl_target_loss \
                    + weight_gen["s2s"]["cycle"] * cycle_s2s_loss + weight_gen["t2t"]["cycle"] * cycle_t2t_loss \
                    + weight_gen["s2s"]["entropy"] * entropy_s2s_loss + weight_gen["t2s"]["entropy"] * entropy_t2s_loss \
                    + weight_gen["s2s"]["classif_vae"] * D_classif_vae_loss \
                    + weight_gen["s2e"]["classif_embedding"] * classif_embedding_loss \
                    + weight_gen["t2e"]["entropy_embedding"] * entropy_embedding_target_loss

        self.FT_G_loss = weight_gen["s2t"]["feat_gan"] * G_FT_s2t_loss \
                       + weight_gen["s2t"]["feat_classif"] * FT_s2t_classif_loss 

        self.FT_D_loss = weight_disc["s2t"]["feat_gan"] * D_FT_s2t_loss

        self.init_G_loss = weight_init["vae_rec_straight"] * (vae_rec_s2s_loss + vae_rec_t2t_loss) \
                         + weight_init["vae_rec_twist"] * (vae_rec_s2t_loss + vae_rec_t2s_loss) \
                         + weight_init["vae_kl"] * (vae_kl_source_loss + vae_kl_target_loss) \
                         
        self.init_D_loss = weight_init["disc_classif_source"] * D_classif_source_loss

        self.init_FS_loss = weight_init["fs_classif_source"] * FS_source_classif_loss
                      
        # ===================== Summary variables =====================
                      
        tf.summary.scalar("entropy_t2s_loss", entropy_t2s_loss)
        tf.summary.scalar("entropy_s2s_loss", entropy_s2s_loss)
        tf.summary.scalar("vae_rec_s2s_loss", vae_rec_s2s_loss)
        tf.summary.scalar("vae_rec_t2t_loss", vae_rec_t2t_loss)
        tf.summary.scalar("vae_rec_s2t_loss", vae_rec_s2t_loss)
        tf.summary.scalar("vae_rec_t2s_loss", vae_rec_t2s_loss)
        tf.summary.scalar("vae_kl_source_loss", vae_kl_source_loss)
        tf.summary.scalar("vae_kl_target_loss", vae_kl_target_loss)
        tf.summary.scalar("D_classif_source_loss", D_classif_source_loss)
        tf.summary.scalar("D_classif_vae_loss", D_classif_vae_loss)
        tf.summary.scalar("feat_matching_t2s_loss", feat_matching_t2s_loss)
        tf.summary.scalar("feat_matching_s2t_loss", feat_matching_s2t_loss)
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
        tf.summary.scalar("D_FT_s2t_loss", D_FT_s2t_loss)
        tf.summary.scalar("G_FT_s2t_loss", G_FT_s2t_loss)
        tf.summary.scalar("FS_source_classif_loss", FS_source_classif_loss)
        tf.summary.scalar("FT_s2t_classif_loss", FT_s2t_classif_loss)
        tf.summary.scalar("classif_embedding_loss", classif_embedding_loss)
        tf.summary.scalar("entropy_embedding_target_loss", entropy_embedding_target_loss)
        tf.summary.scalar("semantic_s2t_loss", semantic_s2t_loss)
        tf.summary.scalar("semantic_t2s_loss", semantic_t2s_loss)
        tf.summary.scalar("G_loss", self.G_loss)
        tf.summary.scalar("D_loss", self.D_loss)
        tf.summary.scalar("accuracy", self.accuracy)
        tf.summary.scalar("accuracy_mean", self.accuracy_mean)
        tf.summary.scalar("FS_source_accuracy", self.FS_source_accuracy)
        tf.summary.scalar("FT_target_accuracy", self.FT_target_accuracy)
        tf.summary.scalar("FT_s2t_accuracy", self.FT_s2t_accuracy)
        tf.summary.scalar("latent_source_accuracy", self.latent_source_accuracy)
        tf.summary.scalar("latent_target_accuracy", self.latent_target_accuracy)

        self.losses_summary = tf.summary.merge_all()
        
        D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='source/discriminator') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target/discriminator') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='source/feature_discriminator') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target/feature_discriminator')

        G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='source/encoder') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target/encoder') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='source/decoder') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target/decoder') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='source/latent_classifier') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target/latent_classifier') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder') \
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')

        FT_G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target/feature_encoder')
        FT_D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target/feature_discriminator')
        FS_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='source/feature_encoder')
                 
        wc = config_train["weight_clipping"]
        self.clip_D = [p.assign(tf.clip_by_value(p, -wc, wc)) for p in D_vars]

        gen = config_train["gen_opti"]
        disc = config_train["disc_opti"]
        lr_gen = config_train["lr_gen"]
        lr_disc = config_train["lr_disc"]
        lr_ft_gen = config_train["lr_ft_gen"]
        lr_ft_disc = config_train["lr_ft_disc"]
        lr_fs = config_train["lr_fs"]

        self.D_solver = self.tf_optimizer[gen_optimizer](learning_rate=float(lr_disc)).minimize(self.D_loss, var_list=D_vars)
        self.G_solver = self.tf_optimizer[disc_optimizer](learning_rate=float(lr_gen)).minimize(self.G_loss, var_list=G_vars)
        self.FT_G_solver = self.tf_optimizer[gen_optimizer](learning_rate=float(lr_ft_gen)).minimize(self.FT_G_loss, var_list=FT_G_vars)
        self.FT_D_solver = self.tf_optimizer[disc_optimizer](learning_rate=float(lr_ft_disc)).minimize(self.FT_D_loss, var_list=FT_D_vars)

        self.init_D_solver = self.tf_optimizer[gen_optimizer](learning_rate=float(lr_disc)).minimize(self.init_D_loss, var_list=D_vars)
        self.init_G_solver = self.tf_optimizer[gen_optimizer](learning_rate=float(lr_gen)).minimize(self.init_G_loss, var_list=G_vars)
        self.init_FS_solver = self.tf_optimizer[fs_optimizer](learning_rate=float(lr_fs)).minimize(self.init_FS_loss, var_list=FS_vars)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)

        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter(summary_dir["train"], self.sess.graph)
        self.test_writer = tf.summary.FileWriter(summary_dir["test"], self.sess.graph)
        self.save_writer = tf.summary.FileWriter(summary_dir["save"])
    
    def train(self, X_source, X_target, Y_source, Y_target, summary_dir):
        config_train = self.config["train"]
        nb_iter = config_train["nb_iter"]
        nb_iter_init = config_train["init"]["iter"]
        nb_iter_g = config_train["nb_iter_g"]
        nb_iter_d = config_train["nb_iter_d"]
        nb_iter_ft_g = config_train["nb_iter_ft_g"]
        nb_iter_ft_d = config_train["nb_iter_ft_d"]
        nb_iter_init_fs = config_train["nb_iter_init_fs"]
        batch_size = config_train["batch_size"]
        save_every = config_train["save_every"]
        test_every = config_train["test_every"]
        
        for self.iter in range(self.iter, nb_iter):
            if self.iter < nb_iter_init:
                D_solver = self.init_D_solver
                G_solver = self.init_G_solver
            else:
                D_solver = self.D_solver
                G_solver = self.G_solver

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

                self.sess.run(D_solver, feed_dict=feed_dict)
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
                             
                self.sess.run(G_solver, feed_dict=feed_dict)

            for k in range(nb_iter_ft_d):
                if self.iter >= nb_iter_init:
                    idx_sample_source = np.random.choice(len(X_source), batch_size)
                    sample_source = X_source[idx_sample_source]
                    sample_source_labels = Y_source[idx_sample_source]
                    sample_target = X_target[np.random.choice(len(X_target), batch_size)]
                    
                    feed_dict = {self.ipt_source: sample_source, 
                                self.ipt_target: sample_target,
                                self.labels_source: sample_source_labels,
                                self.is_train: True}

                    self.sess.run(self.FT_D_solver, feed_dict=feed_dict)

            for k in range(nb_iter_ft_g):
                if self.iter >= nb_iter_init:
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
                                
                    self.sess.run(self.FT_G_solver, feed_dict=feed_dict)
            
            for k in range(nb_iter_init_fs):
                if self.iter < nb_iter_init:
                    idx_sample_source = np.random.choice(len(X_source), batch_size)
                    sample_source = X_source[idx_sample_source]
                    sample_source_labels = Y_source[idx_sample_source]
                    
                    feed_dict = {self.ipt_source: sample_source, 
                                self.labels_source: sample_source_labels, 
                                self.is_train: True}

                    self.sess.run(self.init_FS_solver, feed_dict=feed_dict)
            
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
            print("Saving 'generated-images/iter_{:05d}_image_{:02d}.png'".format(self.iter, index), end="\r")
            plot_images(index, X_source, X_target, X_s2s, X_t2t, X_s2t, X_t2s, X_cycle_s2s, X_cycle_t2t, Y_source_predict, Y_target_predict)
            plt.savefig(os.path.join(images_dir, "iter_{:05d}_image_{:02d}.png".format(self.iter, index)))

            # buf = io.BytesIO()
            # plt.savefig(buf, format='png')
            # buf.seek(0)
            # image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
            # image = tf.expand_dims(image, 0)
            # summary_op = tf.summary.image("image_{}_{}".format(self.iter, index), image)
            # summary = self.sess.run(summary_op)
            # self.save_writer.add_summary(summary, global_step=self.iter)
            
    def save_model(self, model_dir):
        self.saver.save(self.sess, os.path.join(model_dir, "model.ckpt"))
