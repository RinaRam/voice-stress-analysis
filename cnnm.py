import librosa
import matplotlib.pyplot as plt
import librosa.display
from IPython.display import Audio
import numpy as np
from sklearn.dummy import DummyClassifier
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
from sklearn.metrics import confusion_matrix
import IPython.display as ipd 
import os 
import sys
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import tensorflow.keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization, Dense
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report


def male_n():
    x, sr = librosa.load('/content/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav')
    plt.figure(figsize=(8, 4))
    librosa.display.waveplot(x, sr=sr)
    plt.title('Waveplot - Male Neutral')
    plt.savefig('Waveplot_MaleNeutral.png')
    librosa.output.write_wav('content/ipd.Audio Files/MaleNeutral.wav', x, sr)
    Audio(data=x, rate=sr)
    spectrogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128,fmax=8000) 
    spectrogram = librosa.power_to_db(spectrogram)
    librosa.display.specshow(spectrogram, y_axis='mel', fmax=8000, x_axis='time')
    plt.title('Mel Spectrogram - Male Neutral')
    plt.savefig('MelSpec_MaleNeutral.png')
    plt.colorbar(format='%+2.0f dB')
    
    
def fem_c():
    x, sr = librosa.load('/content/Audio_Speech_Actors_01-24/Actor_02/03-01-02-01-01-01-02.wav')
    plt.figure(figsize=(8, 4))
    librosa.display.waveplot(x, sr=sr)
    plt.title('Waveplot - Female Calm')
    plt.savefig('Waveplot_FemaleCalm.png')
    librosa.output.write_wav('content/ipd.Audio Files/FemaleCalm.wav', x, sr)
    Audio(data=x, rate=sr)
    spectrogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128,fmax=8000) 
    spectrogram = librosa.power_to_db(spectrogram)
    librosa.display.specshow(spectrogram, y_axis='mel', fmax=8000, x_axis='time')
    plt.title('Mel Spectrogram - Female Calm')
    plt.savefig('MelSpec_FemaleCalm.png')
    plt.colorbar(format='%+2.0f dB')
    
    
def male_h():
    x, sr = librosa.load('/content/Audio_Speech_Actors_01-24/Actor_03/03-01-03-01-01-01-03.wav')
    plt.figure(figsize=(8, 4))
    librosa.display.waveplot(x, sr=sr)
    plt.title('Waveplot - Male Happy')
    plt.savefig('Waveplot_MaleHappy.png')
    librosa.output.write_wav('content/ipd.Audio Files/MaleHappy.wav', x, sr)
    Audio(data=x, rate=sr)
    x = librosa.feature.melspectrogram(y=x, sr=sr,n_mels=128,fmax=8000) 
    x = librosa.power_to_db(x)
    librosa.display.specshow(x, y_axis='mel', fmax=8000, x_axis='time')
    plt.title('Mel Spectrogram - Male Happy')
    plt.savefig('MelSpec_MaleHappy.png')
    plt.colorbar(format='%+2.0f dB')

    
def fem_s():
    x, sr = librosa.load('/content/Audio_Speech_Actors_01-24/Actor_04/03-01-04-01-01-01-04.wav')
    plt.figure(figsize=(8, 4))
    librosa.display.waveplot(x, sr=sr)
    plt.title('Waveplot - Female Sad')
    plt.savefig('Waveplot_FemaleSad.png')
    librosa.output.write_wav('content/ipd.Audio Files/FemaleSad.wav', x, sr)
    Audio(data=x, rate=sr)
    x = librosa.feature.melspectrogram(y=x, sr=sr,n_mels=128,fmax=8000) 
    y = librosa.power_to_db(x)
    librosa.display.specshow(y, y_axis='mel', fmax=8000, x_axis='time')
    plt.title('Mel Spectrogram - Female Sad')
    plt.savefig('MelSpec_FemaleSad.png')
    plt.colorbar(format='%+2.0f dB')
    
    
def male_a():
    x, sr = librosa.load('/content/Audio_Speech_Actors_01-24/Actor_05/03-01-05-01-01-01-05.wav')
    plt.figure(figsize=(8, 4))
    librosa.display.waveplot(x, sr=sr)
    plt.title('Waveplot - Male Angry')
    plt.savefig('Waveplot_MaleAngry.png')
    librosa.output.write_wav('content/ipd.Audio Files/MaleAngry.wav', x, sr)
    Audio(data=x, rate=sr)
    x = librosa.feature.melspectrogram(y=x, sr=sr,n_mels=128,fmax=8000) 
    y = librosa.power_to_db(x)
    librosa.display.specshow(y, y_axis='mel', fmax=8000, x_axis='time')
    plt.title('Mel Spectrogram - Male Angry')
    plt.savefig('MelSpec_MaleAngry.png')
    plt.colorbar(format='%+2.0f dB')
    
    
def fem_f():
    x, sr = librosa.load('/content/Audio_Speech_Actors_01-24/Actor_06/03-01-06-01-01-01-06.wav')
    plt.figure(figsize=(8, 4))
    librosa.display.waveplot(x, sr=sr)
    plt.title('Waveplot - Female Fearful')
    plt.savefig('Waveplot_FemaleFearful.png')
    librosa.output.write_wav('content/ipd.Audio Files/FemaleFearful.wav', x, sr)
    Audio(data=x, rate=sr)
    x = librosa.feature.melspectrogram(y=x, sr=sr,n_mels=128,fmax=8000) 
    y = librosa.power_to_db(x)
    librosa.display.specshow(y, y_axis='mel', fmax=8000, x_axis='time')
    plt.title('Mel Spectrogram - Female Fearful')
    plt.savefig('MelSpec_FemaleFearful.png')
    plt.colorbar(format='%+2.0f dB')
    
def male_d():
    x, sr = librosa.load('/content/Audio_Speech_Actors_01-24/Actor_07/03-01-07-01-01-01-07.wav')
    plt.figure(figsize=(8, 4))
    librosa.display.waveplot(x, sr=sr)
    plt.title('Waveplot - Male Disgust')
    plt.savefig('Waveplot_MaleDisgust.png')
    librosa.output.write_wav('content/ipd.Audio Files/MaleDisgust.wav', x, sr)
    Audio(data=x, rate=sr)
    x = librosa.feature.melspectrogram(y=x, sr=sr,n_mels=128,fmax=8000) 
    y = librosa.power_to_db(x)
    librosa.display.specshow(y, y_axis='mel', fmax=8000, x_axis='time')
    plt.title('Mel Spectrogram - Male Disgust')
    plt.savefig('MelSpec_MaleDisgust.png')
    plt.colorbar(format='%+2.0f dB')

    
def fem_surp():
    x, sr = librosa.load('/content/Audio_Speech_Actors_01-24/Actor_08/03-01-08-01-01-01-08.wav')
    plt.figure(figsize=(8, 4))
    librosa.display.waveplot(x, sr=sr)
    plt.title('Waveplot - FemaleSurprised')
    plt.savefig('Waveplot_FemaleSurprised.png')
    librosa.output.write_wav('content/ipd.Audio Files/FemaleSurprised.wav', x, sr)
    Audio(data=x, rate=sr)
    x = librosa.feature.melspectrogram(y=x, sr=sr,n_mels=128,fmax=8000) 
    y = librosa.power_to_db(x)
    librosa.display.specshow(y, y_axis='mel', fmax=8000, x_axis='time')
    plt.title('Mel Spectrogram - Female Surprised')
    plt.savefig('MelSpec_FemaleSurprised.png')
    plt.colorbar(format='%+2.0f dB')
    

def labl_aud_fls():
    audio = "/content/Audio_Speech_Actors_01-24/"
    actor_folders = os.listdir(audio)
    actor_folders.sort() 
    emotion = []
    gender = []
    actor = []
    file_path = []
    for i in actor_folders:
        filename = os.listdir(audio + i)
        for f in filename:
            part = f.split('.')[0].split('-')
            emotion.append(int(part[2]))
            actor.append(int(part[6]))
            bg = int(part[6])
            if bg%2 == 0:
                bg = "female"
            else:
                bg = "male"
            gender.append(bg)
            file_path.append(audio + i + '/' + f)
    audio_df = pd.DataFrame(emotion)
    audio_df = audio_df.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
    audio_df = pd.concat([pd.DataFrame(gender),audio_df,pd.DataFrame(actor)],axis=1)
    audio_df.columns = ['gender','emotion','actor']
    audio_df = pd.concat([audio_df,pd.DataFrame(file_path, columns = ['path'])],axis=1)
    pd.set_option('display.max_colwidth', -1)
    audio_df.sample(10)
    audio_df.emotion.value_counts().plot(kind='bar')
    audio_df.to_csv('content/audio.csv')
    
    
def ftr_extr():
    df = pd.DataFrame(columns=['mel_spectrogram'])
    for index, path in enumerate(audio_df.path):
        X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)
        spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000) 
        db_spec = librosa.power_to_db(spectrogram)
        log_spectrogram = np.mean(db_spec, axis = 0)
        
        mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
        mfcc = np.mean(mfcc,axis=0)
        chroma = librosa.feature.chroma_stft(y=X, sr=sample_rate)
        chroma = np.mean(chroma, axis = 0)
        contrast = librosa.feature.spectral_contrast(y=X, sr=sample_rate)
        contrast = np.mean(contrast, axis= 0)
        which the signal changes from positive to negative or back - separation of voiced andunvoiced speech.)
        zcr = librosa.feature.zero_crossing_rate(y=X)
        zcr = np.mean(zcr, axis= 0)

        df.loc[index] = [log_spectrogram]
    print(len(df))
    df.head()
    df_combined = pd.concat([audio_df,pd.DataFrame(df['mel_spectrogram'].values.tolist())],axis=1)
    df_combined = df_combined.fillna(0)
    df_combined.drop(columns='path',inplace=True)
    df_combined.head()
    
    
def data_prep():
    train,test = train_test_split(df_combined, test_size=0.2, random_state=0, stratify=df_combined[['emotion','gender','actor']])
    X_train = train.iloc[:, 3:]
    y_train = train.iloc[:,:2].drop(columns=['gender'])
    print(X_train.shape)
    X_test = test.iloc[:,3:]
    y_test = test.iloc[:,:2].drop(columns=['gender'])
    print(X_test.shape)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    lb = LabelEncoder()
    y_train = to_categorical(lb.fit_transform(y_train))
    y_test = to_categorical(lb.fit_transform(y_test))
    print(y_test[0:10])
    print(lb.classes_)
    X_train = X_train[:,:,np.newaxis]
    X_test = X_test[:,:,np.newaxis]
    X_train.shape
   

def base_model():
    print(X_train.shape)
    print(X_test.shape)
    dummy_clf = DummyClassifier(strategy="stratified")
    dummy_clf.fit(X_train, y_train)
    DummyClassifier(strategy='stratified')
    dummy_clf.predict(X_test)
    dummy_clf.score(X_test, y_test)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    clf.predict(X_test)
    clf.score(X_test, y_test)
    

def build_model():
    model = Sequential()
    model.add(Conv1D(64, kernel_size=(10), activation='relu', input_shape=(X_train.shape[1],1)))
    model.add(Conv1D(128, kernel_size=(10),activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Dropout(0.4))
    model.add(Conv1D(128, kernel_size=(10),activation='relu'))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(8, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.Adam(lr=0.001),metrics=['accuracy'])
    model.summary()
    checkpoint = ModelCheckpoint("best_initial_model.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', period=1, save_weights_only=True)
    model_history = model.fit(X_train, y_train,batch_size=32, epochs=40, validation_data=(X_test, y_test),callbacks=[checkpoint])
    return model


def plot():
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Initial_Model_Accuracy.png')
    plt.show()
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Initial_Model_loss.png')
    plt.show()
