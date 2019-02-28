import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import argparse
from loss_functions import cross_entropy_balanced, pixel_error
from model.rcf_model_vgg import vgg_rcf


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description='Keras Training')
    # ========= paths for training
    ap.add_argument("-npath", "--npy_path", required=True,
                    help="path to npy. files to train")
    ap.add_argument("-mpath", "--model_path", required=True,
                    help="path to save the output model")
    ap.add_argument("-lpath", "--log_path", required=True,
                    help="path to save the 'log' files")
    ap.add_argument("-name","--model_name", required=True,
                    help="output of model name")
    # ========= parameters for training
    ap.add_argument("-r", "--rows", required=True, type=int,
                    help="shape of rows of input image")
    ap.add_argument("-c", "--cols", required=True, type=int,
                    help="shape of cols of input image")
    ap.add_argument('-bs', '--batch_size', default=1, type=int,
                    help='batch size')
    ap.add_argument('-ep', '--epoch', default=1, type=int,
                    help='epoch')
    ap.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    args = vars(ap.parse_args())
    return args


def train(args):
    X_train = np.load(args["npy_path"] + 'X_train_ori.npy')
    X_val = np.load(args["npy_path"] + 'X_val_ori.npy')
    y_train = np.load(args["npy_path"] + 'y_train_concat.npy')
    y_val = np.load(args["npy_path"] + 'y_val_concat.npy')
    model_vgg_rcf = vgg_rcf(input_shape=(args["rows"], args["cols"], 3))

    model_vgg_rcf.summary()
    lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1, min_lr=1e-4)
    checkpointer = ModelCheckpoint(args["model_path"] + args["model_name"], verbose=1, save_best_only=True)
    callback_list = [lr_decay, checkpointer]

    # optimizer = SGD(lr=1e-3, momentum=args["momentum"], nesterov=False)
    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)

    model_vgg_rcf.compile(loss={'o1': cross_entropy_balanced,
                                     'o2': cross_entropy_balanced,
                                     'o3': cross_entropy_balanced,
                                     'o4': cross_entropy_balanced,
                                     'o5': cross_entropy_balanced,
                                     'ofuse': cross_entropy_balanced,
                                     },
                                     metrics={'ofuse': pixel_error},
                                     optimizer=optimizer)

    RCF = model_vgg_rcf.fit(X_train, [y_train, y_train, y_train, y_train, y_train, y_train],
                                 validation_data=(X_val, [y_val, y_val, y_val, y_val, y_val, y_val]),
                                 batch_size=args["batch_size"], epochs=args["epoch"], callbacks=callback_list, verbose=1)


if __name__ == "__main__":
    args = args_parse()
    train(args)