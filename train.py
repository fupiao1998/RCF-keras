import numpy as np
from keras.optimizers import Adam, SGD
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import argparse
from loss_functions import cross_entropy_balanced, pixel_error, cross_entropy_loss_RCF
from model.hed_model_resnet import res_hed
from model.rcf_model_vgg import vgg_rcf
from model.hed_model_vgg import vgg_hed
from model.rcf_model_large_resnet import resnet_rcf_large
from model.rcf_model_resnext import resnext_rcf
from model.rcf_model_resnet import resnet_rcf

'''
-npath C:\\Users\\dell\\Downloads\\BSR_bsds500\\BSR\\BSDS500\\data\\
-mpath D:\\all-PythonCodes\\RCFs\\RCF-keras\\
-lpath D:\\all-PythonCodes\\RCFs\\RCF-keras\\log\\
-name resnet_rcf_test
-r 320
-c 480
'''

'''
-npath D:\\all-PythonCodes\\RCFs\\RCF-keras\\building_data\\111\\
-mpath D:\\all-PythonCodes\\RCFs\\RCF-keras\\
-lpath D:\\all-PythonCodes\\RCFs\\RCF-keras\\log\\
-name resnet_rcf_test.h5
-r 256
-c 256
'''

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
    ap.add_argument("-p", "-pretrain", default=0, required=True, type=int,
                    help="load pre-train model or not")
    ap.add_argument("-r", "--rows", required=True, type=int,
                    help="shape of rows of input image")
    ap.add_argument("-c", "--cols", required=True, type=int,
                    help="shape of cols of input image")
    ap.add_argument('-bs', '--batch_size', default=2, type=int,
                    help='batch size')
    ap.add_argument('-ep', '--epoch', default=1000, type=int,
                    help='epoch')
    ap.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    args = vars(ap.parse_args())
    return args


def train(args):
    X_train = np.load(args["npy_path"] + 'X_train.npy')
    # X_val = np.load(args["npy_path"] + 'X_val.npy')
    y_train = np.load(args["npy_path"] + 'y_train.npy')
    # y_val = np.load(args["npy_path"] + 'y_val.npy')
    if args["pretrain"]:
        model_vgg_rcf = load_model(args["model_path"] + args["model_name"],
                       custom_objects={'cross_entropy_balanced': cross_entropy_balanced, 'pixel_error': pixel_error})
    else:
        model = resnext_rcf(input_shape=(args["rows"], args["cols"], 3))

    model.summary()
    lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1, min_lr=1e-4)
    checkpointer = ModelCheckpoint(args["model_path"] + args["model_name"], verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=args["log_path"])
    callback_list = [lr_decay, checkpointer, tensorboard]

    # optimizer = SGD(lr=1e-3, momentum=args["momentum"], nesterov=False)
    optimizer = Adam(lr=1e-2, beta_1=0.9, beta_2=0.999)

    model.compile(loss={'o1': cross_entropy_balanced,
                                'o2': cross_entropy_balanced,
                                'o3': cross_entropy_balanced,
                                'o4': cross_entropy_balanced,
                                'o5': cross_entropy_balanced,
                                'ofuse': cross_entropy_balanced,
                                },
                                metrics={'ofuse': pixel_error},
                                optimizer=optimizer)

    # RCF = model_vgg_rcf.fit(X_train, [y_train, y_train, y_train, y_train, y_train, y_train],
    #                             validation_data=(X_val, [y_val, y_val, y_val, y_val, y_val, y_val]),
    #                             batch_size=args["batch_size"], epochs=args["epoch"],
    #                             callbacks=callback_list, verbose=1)

    RCF = model.fit(X_train, [y_train, y_train, y_train, y_train, y_train, y_train],validation_split=0.2,
                            batch_size=args["batch_size"], epochs=args["epoch"], callbacks=callback_list, verbose=1)


if __name__ == "__main__":
    args = args_parse()
    train(args)