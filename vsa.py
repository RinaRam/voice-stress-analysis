
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Flatten, Dropout, Activation, Conv1D, MaxPooling1D, AveragePooling1D, BatchNormalization, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from tensorflow.keras import regularizers
import os



mylist= os.listdir('Audio_Speech_Actors_01-24/')

feeling_list=[]
for item in mylist:
    if item[6:-16]=='02':
        feeling_list.append('calm')
    elif item[6:-16]=='03':
        feeling_list.append('happy')
    elif item[6:-16]=='04':
        feeling_list.append('sad')
    elif item[6:-16]=='05' :
        feeling_list.append('angry')
    elif item[6:-16]=='06':
        feeling_list.append('fearful')
    elif item[:1]=='a':
        feeling_list.append('angry')
    elif item[:1]=='f':
        feeling_list.append('fearful')
    elif item[:1]=='h':
        feeling_list.append('happy')
    elif item[:1]=='n':
        feeling_list.append('neutral')
    elif item[:2]=='sa':
        feeling_list.append('sad')


def normalize(wav):
    mean = wav.mean(axis=0, dtype=np.float32)
    dispersion = wav.std(axis=0, dtype=np.float32)
    return (wav - mean) / dispersion

def get_train_data(img_train, dir, fast_train=False):
    if fast_train:
        len_train = 10
    else:
        len_train = len(img_train) * 7
    X = np.zeros((len_train, IMG_SIZE, IMG_SIZE))
    y = np.zeros((len_train, FACEPOINTS, 2))
    i = 0
    for filename in img_train:
        pnts = img_train[filename]
        img = imread(os.path.join(dir, filename), as_gray= True)
        new_pnts = np.dstack((pnts[::2], pnts[1::2]))[0]
        X[i], y[i] = resize_img(img, new_pnts, False)
        if fast_train:
            i += 1
            if i == len_train:
                break
        else:
            crop = (int)((np.random.randint(0, 9) / 1000.0) * IMG_SIZE)
            X[i + 6], y[i + 6] = resize_img(img[crop:IMG_SIZE - crop:, crop:IMG_SIZE - crop:], new_pnts - crop, False)
            X[i + 5], y[i + 5] = flip_img(img, new_pnts)
            for j in range(4):
                X[i + j + 1], y[i + j + 1] = random_rotate(img, new_pnts)
            i += 7
    return normalize_img(X), y


def build_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='valid', kernel_initializer='random_uniform', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_initializer='random_uniform'))
    model.add(Activation('elu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_initializer='random_uniform'))
    model.add(Activation('elu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, (2, 2), padding='valid', kernel_initializer='random_uniform'))
    model.add(Activation('elu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(1000))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000))
    model.add(Activation('linear'))
    model.add(Dropout(0.6))
    model.add(Dense(FACEPOINTS * 2))
    model.add(Reshape((FACEPOINTS, 2)))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

def train_detector(train_gt, train_dir, fast_train = True):
    if fast_train:
        batches = 3
        epochs = 1
    else:
        batches = BATCHES
        epochs = EPOCHS
    X_train, y_train = get_train_data(train_gt, train_dir, fast_train)
    model = build_model()
    model.fit(X_train, y_train, batch_size = batches, epochs = epochs, validation_split = 0.1)
    model.save('vsa_model.hdf5')

def detect(model, test_dir):
    filenames = os.listdir(test_dir)
    len_test = len(filenames)

    X = np.zeros((len_test, IMG_SIZE, IMG_SIZE))

    transforms = np.zeros(len_test)
    for i in range(len(filenames)):
        filename = filenames[i]
        wav = ...(os.path.join(test_dir, filename))
        wav, _, transform = chng(img, None, True)
        X[i] = wav
        transforms[i] = transform
    X_test = normalize(X)
    y_pred = model.predict(X_test)
    ans = {}
    for i in range(len(y_pred)):
        y_pred[i] *= transforms[i]
        ans[filenames[i]] = np.array(y_pred[i]).flatten().tolist()
    return ans
