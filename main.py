import os

import mne
import numpy as np
import pandas as pd
import pyts.image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import EEGModels
from EEGModels import EEGNet


def find_set_files(path):
    set_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.set'):
                set_files.append(os.path.join(root, file))
    return set_files

subjects_data_path = 'C:\\Users\egusa\dataseteeg\derivatives'

labels_path = 'C:\\Users\egusa\dataseteeg\participants.tsv'

labels_dataframe = pd.read_csv(labels_path, sep='\t')

print(labels_dataframe)

path_to_search = subjects_data_path
set_files = find_set_files(path_to_search)
file_list = []
for file in set_files:
    file_list.append(file)
print(file_list)

n_files = len(file_list)
print(n_files)

channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz',
                 'Cz', 'Pz']

data_list = []

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

downsampling_ratio = 50

times_list = []
for i, file in enumerate(file_list):
    raw = mne.io.read_raw_eeglab(file, preload=True)

    data = raw.get_data(picks=channel_names)

    transposed_data = np.transpose(data)
    times = raw.times

    data = data*100

    data = pd.DataFrame(transposed_data, columns=channel_names)
    times = pd.DataFrame(np.transpose(times), columns=['times'])

    #Downsampling
    data = data.groupby(data.index // downsampling_ratio).median()
    times = times.groupby(times.index // downsampling_ratio).first()

    data = data.iloc[:151]

    data_list.append(data)
    times_list.append(times)

time_points = np.array(times_list[0]).transpose()

# GAF conversion
#
# images_array = []
# for sub in data_list:
#     images = []
#     for column_name in sub.columns:
#         x = np.array([sub[column_name]])
#
#         gaf = pyts.image.GramianAngularField(image_size=400, sample_range=(0, 1))
#
#         X_gaf = gaf.fit_transform(x)
#
#         images.append(X_gaf)
#     images_array.append(images)
#
# normalized_images_array = []
#
# scaler = MinMaxScaler(feature_range=(0, 1))
#
# for images in images_array:
#     normalized_images = []
#     for image in images:
#         normalized_images.append(scaler.fit_transform(image[0]))
#     normalized_images_array.append(normalized_images)
#
# x = normalized_images_array

x = data_list

labels_dataframe = labels_dataframe["Group"]


from tensorflow.keras.utils import to_categorical

new_y = labels_dataframe.map({'F': 2, 'A': 1, 'C': 0})

new_df = pd.DataFrame(new_y)

new_df = to_categorical(new_df)
print(new_df)


from sklearn.model_selection import train_test_split

new_x = np.asarray(x)
new_x = new_x.reshape(88, 19, 151)

x_train, x_test, y_train, y_test = train_test_split(new_x, new_df, test_size=0.2, random_state=42)
# Дополнительное разделение обучающей выборки на обучающую и валидационную
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

kernels, chans, samples = 1, 19, 151
x_train = np.asarray(x_train, np.float32)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test, np.float32)
y_test = np.asarray(y_test)
x_val = np.asarray(x_val, np.float32)
y_val = np.asarray(y_val)

############################# EEGNet portion ##################################
from tensorflow.keras import utils as np_utils

# convert data to NHWC (trials, channels, samples, kernels) format. Data
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
X_train = x_train.reshape(x_train.shape[0], chans, samples, kernels)
X_validate = x_val.reshape(x_val.shape[0], chans, samples, kernels)
X_test = x_test.reshape(x_test.shape[0], chans, samples, kernels)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
# model configurations may do better, but this is a good starting point)
model = EEGNet(nb_classes=3, Chans=chans, Samples=samples,
               dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
               dropoutType='Dropout')

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# count number of parameters in the model
numParams = model.count_params()

###############################################################################
# if the classification task was imbalanced (significantly more trials in one
# class versus the others) you can assign a weight to each class during
# optimization to balance it out. This data is approximately balanced so we
# don't need to do this, but is shown here for illustration/completeness.
###############################################################################

# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
# the weights all to be 1
class_weights = {0: 1, 1: 1, 2: 1}

################################################################################
# fit the model. Due to very small sample sizes this can get
# pretty noisy run-to-run, but most runs should be comparable to xDAWN +
# Riemannian geometry classification (below)
################################################################################
fittedModel = model.fit(X_train, y_train, batch_size=16, epochs=600,
                        verbose=2, validation_data=(X_validate, y_val))

###############################################################################
# can alternatively used the weights provided in the repo. If so it should get
# you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
# system.
###############################################################################

# WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5
# model.load_weights(WEIGHTS_PATH)

###############################################################################
# make prediction on test set.
###############################################################################

probs = model.predict(X_test)
preds = probs.argmax(axis=-1)
acc = np.mean(preds == y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

############################# PyRiemann Portion ##############################

# code is taken from PyRiemann's ERP sample script, which is decoding in
# the tangent space with a logistic regression
# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

n_components = 2  # pick some components

# set up sklearn pipeline
clf = make_pipeline(XdawnCovariances(n_components),
                    TangentSpace(metric='riemann'),
                    LogisticRegression())

preds_rg = np.zeros(len(y_test))

# reshape back to (trials, channels, samples)
X_train = X_train.reshape(X_train.shape[0], chans, samples)
X_test = X_test.reshape(X_test.shape[0], chans, samples)

# train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
# labels need to be back in single-column format
clf.fit(X_train, y_train.argmax(axis=-1))
preds_rg = clf.predict(X_test)

# Printing the results
acc2 = np.mean(preds_rg == y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc2))

