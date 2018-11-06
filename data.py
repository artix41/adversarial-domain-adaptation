import os
import numpy as np
from skimage import color, transform
from sklearn.preprocessing import LabelBinarizer
from torchvision import datasets
from tensorflow.examples.tutorials.mnist import input_data

from utils import normalize

def transform_mnist(X, use_inverse=False):
    X = X.reshape(len(X), 28, 28)
    if use_inverse:
        X = np.concatenate([X, 1-X], axis=0)
    X = color.gray2rgb(transform.resize(X, [X.shape[0], 32,32]))
    X = normalize(X, 1)
    X = X.reshape(len(X), 32, 32, 3)
    
    return X
    
def transform_svhn(X):
    X = np.transpose(X, (0,2,3,1))
    X = normalize(X, 256)
    X = X.reshape(len(X), 32, 32, 3)
    
    return X
    
def load_mnist(config, verbose=2):
    if verbose >= 2:
        print("[*] Loading MNIST")
    prep_train_file = os.path.join(config["path"], "mnist-train.npy")
    prep_train_labels_file = os.path.join(config["path"], "mnist-train-labels.npy")
    prep_test_file = os.path.join(config["path"], "mnist-test.npy")
    prep_test_labels_file = os.path.join(config["path"], "mnist-test-labels.npy")

    if os.path.exists(prep_train_file) and os.path.exists(prep_test_file):
        if verbose >= 2:
            print("   [+] Preprocessed data found")
        X_train = np.load(prep_train_file)
        Y_train = np.load(prep_train_labels_file)
        X_test = np.load(prep_test_file)
        Y_test = np.load(prep_test_labels_file)
    else:
        print("Load train/test")
        mnist = input_data.read_data_sets(config["path"], one_hot=False)
        print("Transform train")
        X_train = transform_mnist(mnist.train.images, use_inverse=config["use_inverse"])
        print("Transform test")
        X_test = transform_mnist(mnist.test.images, use_inverse=config["use_inverse"])

        lb = LabelBinarizer()
        Y_train = lb.fit_transform(mnist.train.labels)
        Y_test = lb.fit_transform(mnist.test.labels)
        
        print("Save train")
        np.save(prep_train_file, X_train)
        np.save(prep_train_labels_file, Y_train)
        print("Save test")
        np.save(prep_test_file, X_test)
        np.save(prep_test_labels_file, Y_test)
    
    return X_train, X_test, Y_train, Y_test
    
def load_svhn(config, verbose=2):
    if verbose >= 2:
        print("[*] Loading SVHN")
    prep_train_file = os.path.join(config["path"], "svhn-train.npy")
    prep_train_labels_file = os.path.join(config["path"], "svhn-train-labels.npy")
    prep_test_file = os.path.join(config["path"], "svhn-test.npy")
    prep_test_labels_file = os.path.join(config["path"], "svhn-test-labels.npy")
    
    if os.path.exists(prep_train_file) and os.path.exists(prep_test_file):
        if verbose >= 2:
            print("   [+] Preprocessed data found")
        X_train = np.load(prep_train_file)
        X_test = np.load(prep_test_file)
        Y_train = np.load(prep_train_labels_file)
        Y_test = np.load(prep_test_labels_file)
    else:
        print("Load train")
        svhn_train = datasets.SVHN(root=config["path"], download=False, split="train") # split="extra" works also
        print("Load test")
        svhn_test = datasets.SVHN(root=config["path"], download=False, split="test")
        print("Transform train")
        X_train = transform_svhn(svhn_train.data)
        print("Transform test")
        X_test = transform_svhn(svhn_test.data)
        
        lb = LabelBinarizer()
        Y_train = lb.fit_transform(svhn_train.labels.flatten() % 10)
        Y_test = lb.fit_transform(svhn_test.labels.flatten() % 10)
        
        print("Save train")
        np.save(prep_train_file, X_train)
        np.save(prep_train_labels_file, Y_train)
        print("Save test")
        np.save(prep_test_file, X_test)
        np.save(prep_test_labels_file, Y_test)

    
    return X_train, X_test, Y_train, Y_test
