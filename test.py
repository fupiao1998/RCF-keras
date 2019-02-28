import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from loss_functions import cross_entropy_balanced, pixel_error
import argparse

def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description='Keras Training')
    # ========= paths for training
    ap.add_argument("-npath", "--npy_path", required=True,
                    help="path to npy. files to train")
    ap.add_argument("-mpath", "--model_path", required=True,
                    help="path to save the output model")
    ap.add_argument("-name","--model_name", required=True,
                    help="output of model name")
    ap.add_argument("-r", "--rows", required=True, type=int, default=320,
                    help="shape of rows of input image")
    ap.add_argument("-c", "--cols", required=True, type=int, default=480,
                    help="shape of cols of input image")
    args = vars(ap.parse_args())
    return args


def test(args):
    X_train = np.load(args["npy_path"] + 'X_train_ori.npy')
    X_test = np.load(args["npy_path"] + 'X_test_ori.npy')
    X_val = np.load(args["npy_path"] + 'X_val_ori.npy')
    y_train = np.load(args["npy_path"] + 'y_train_concat.npy')
    y_test = np.load(args["npy_path"] + 'y_test_concat.npy')
    y_val = np.load(args["npy_path"] + 'y_val_concat.npy')
    model = load_model(args["model_path"] + args["model_name"],
                       custom_objects={'cross_entropy_balanced': cross_entropy_balanced, 'pixel_error': pixel_error})
    # test all images from test.npy
    print(len(X_train))
    for i in range(200):
        y_pred = model.predict(X_train[i].reshape((-1, 320, 480, 3)))[-1]

        y_pred = y_pred.reshape((320, 480))
        plt.figure(figsize=(25, 16))
        plt.subplot(1, 3, 1)
        plt.imshow(X_train[i], cmap='binary')
        plt.subplot(1, 3, 2)
        plt.imshow(y_train[i].reshape((320, 480)), cmap='binary')
        plt.subplot(1, 3, 3)
        plt.imshow(y_pred, cmap='binary')
        name = str(i) + '.jpg'
        plt.savefig(name)


if __name__ == "__main__":
    args = args_parse()
    test(args)