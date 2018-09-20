import sys
import os
import tensorflow as tf

from ADA import ADA
from data import load_mnist, load_svhn

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def experiment(config, output_dir, load=False, verbose=2, seed=42):
    tf.set_random_seed(seed)
    
    net_config = config["networks"]
    train_config = config["train"]
    test_config = config["test"]
    data_config = config["data"]
    results_config = config["results"]

    model_dir = os.path.join(output_dir, "model")
    images_dir = os.path.join(output_dir, "images")
    summary_dir = {"train": os.path.join(output_dir, "summary_train"),
                   "test": os.path.join(output_dir, "summary_test")}
    
    nb_iter = train_config["nb_iter"]
    test_every = train_config["test_every"]
    save_every = train_config["save_every"]
    
    verbose = verbose
    
    # =========== Load data ===========
    
    if verbose >= 1:
        print("[*] Loading data...")
        
    X_source_train, X_source_test, Y_source_train, Y_source_test = load_svhn(data_config["svhn"])
    X_target_train, X_target_test, Y_target_train, Y_target_test = load_mnist(data_config["mnist"])
    
    # =========== Start the session ===========
    
    with tf.Session() as sess:
        if verbose >= 1:
            print("[*] Building model...\n")
        ada = ADA(config, output_dir, sess, verbose=verbose)
        ada.build_model(summary_dir)
        
        # =========== Load the model and create the directories ===========
        
        if load:
            if verbose >= 1:
                print("[*] Loading existing model...")
            ada.load(model_dir)
        else:
            os.makedirs(images_dir)
            os.makedirs(model_dir)
        
        # ============= Training ==============
        
        if verbose >= 1:
            print("---------------------- Training ----------------------\n")

        while ada.iter < nb_iter:
            ada.train(X_source_train, X_target_train, Y_source_train, Y_target_train, summary_dir["train"])
            
            if ada.iter % test_every == 0:
                ada.test(X_source_test, X_target_test, Y_source_test, Y_target_test)
            if ada.iter % save_every == 0:
                ada.save_model(model_dir)
                ada.save_images(X_source_test, X_target_test, images_dir, nb_images=results_config["nb_images"])
            
            ada.iter += 1
            
        # ============= Testing ==============
        
        if verbose >= 1:
            print("---------------------- Testing ----------------------\n")
        
        acc = ada.test(X_source_test, X_target_test, Y_source_test, Y_target_test, test_all=True)
        
        if verbose >= 1:
            print("Final accuracy: {:0.5f}\n".format(acc))
    
