import numpy as np
import os
import cv2

from glob import glob
from tqdm import tqdm
from scipy.io import loadmat
from PIL import Image

DATA_PATH = 'C:\\Users\\dell\\Downloads\\BSR_bsds500\\BSR\\BSDS500\\data\\'
GROUND_PATH = DATA_PATH + '\\groundTruth\\'
ORI_IMAGE_PATH = DATA_PATH + '\\images\\'


def get_label():
    y_train = []
    y_test = []
    y_val = []

    for foldname in tqdm(os.listdir(GROUND_PATH)):
        for img in sorted(os.listdir(os.path.join(GROUND_PATH, foldname))):
            IMG_DIR = os.path.join(GROUND_PATH, foldname, img)
            mat = loadmat(IMG_DIR)
            matdata = mat['groundTruth']

            # rotate to the same size
            data = matdata[0, 1]['Boundaries'][0, 0]
            if data.shape == (481, 321):  # size: H - W
                data = data.transpose()
            data = cv2.resize(data, (480, 320))

            if foldname == 'test':
                y_test.append(data)
            elif foldname == 'val':
                y_val.append(data)
            else:
                y_train.append(data)

    y_train = np.asarray(y_train).reshape((-1, 320, 480, 1)).astype('float32')
    y_test = np.asarray(y_test).reshape((-1, 320, 480, 1)).astype('float32')
    y_val = np.asarray(y_val).reshape((-1, 320, 480, 1)).astype('float32')
    np.save(DATA_PATH + 'y_train.npy', y_train)
    np.save(DATA_PATH + 'y_test.npy', y_test)
    np.save(DATA_PATH + 'y_val.npy', y_val)


def get_images():
    X_train = []
    X_test = []
    X_val = []

    for foldname in tqdm(os.listdir(ORI_IMAGE_PATH)):
        i = 0
        for img in sorted(glob(ORI_IMAGE_PATH + '\\' + foldname + '\\*')):
            data = cv2.imread(img)
            if data.shape == (481, 321, 3):
                data = cv2.resize(data, (320, 480))
                data = Image.fromarray(data)
                data = np.asarray(data.rotate(90, expand=True))
            else:
                data = cv2.resize(data, (480, 320))
            if foldname == 'test':
                X_test.append(data)
            elif foldname == 'val':
                X_val.append(data)
            else:
                X_train.append(data)
            i += 1

    X_train = np.asarray(X_train).astype('float32')
    X_test = np.asarray(X_test).astype('float32')
    X_val = np.asarray(X_val).astype('float32')
    np.save(DATA_PATH + 'X_train.npy', X_train)
    np.save(DATA_PATH + 'X_test.npy', X_test)
    np.save(DATA_PATH + 'X_val.npy', X_val)

print('Load Labels')
get_label()
print('Finished Label Loading')
print('load images')
get_images()
print('Finished Data Loading')


