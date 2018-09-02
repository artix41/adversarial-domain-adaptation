import os
import numpy as np
from skimage import color, transform
from sklearn.preprocessing import LabelBinarizer
from torchvision import datasets

from utils import normalize

def transform_mnist(X):
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
    prep_train_file = os.path.join(config["path"], "mnist-train-prep.npy")
    prep_test_file = os.path.join(config["path"], "mnist-test-prep.npy")
    if os.exists(prep_file) and os.exists(prep_test_file):
        if verbose >= 2:
            print("[+] Preprocessed data found")
        X_train = np.load(prep_train_file)
        X_test = np.load(prep_test_file)
    else:
        mnist = input_data.read_data_sets(folder_mnist, one_hot=False)
        X_train = transform_mnist(mnist.train.images)
        X_test = transform_mnist(mnist.test.images)

        np.save(X_train, prep_train_file)
        np.save(X_test, prep_test_file)
        
    lb = LabelBinarizer()
    Y_train = lb_mnist.fit_transform(mnist.train.labels)
    Y_test = mnist.test.labels
    
    return X_train, X_test, Y_train, Y_test
    
def load_svhn(config, verbose=2):
    if verbose >= 2:
        print("[*] Loading SVHN")
    prep_train_file = os.path.join(config["path"], "svhn-train-prep.npy")
    prep_test_file = os.path.join(config["path"], "svhn-test-prep.npy")
    
    if os.path.exists(prep_train_file) and os.path.exists(prep_test_file):
        if verbose >= 2:
            print("[+] Preprocessed data found")
        X_train = np.load(prep_train_file)
        X_test = np.load(prep_test_file)
    else:
        print("load train")
        svhn_train = datasets.SVHN(root=config["path"], download=False, split="extra")
        print("load test")
        svhn_test = datasets.SVHN(root=config["path"], download=False, split="test")
        print("transform train")
        X_train = transform_svhn(svhn_train.data)
        print("transform test")
        X_test = transform_svhn(svhn_test.data)
        
        print("save train")
        np.save(X_train, prep_train_file)
        print("save test")
        np.save(X_test, prep_test_file)

    lb = LabelBinarizer()
    Y_train = lb.fit_transform(svhn_train.labels.flatten() % 10)
    Y_test = lb.fit_transform(svhn_test.labels.flatten() % 10)
    
    return X_train, X_test, Y_train, Y_test
