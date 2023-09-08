# Importing Libraries
import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from glob2 import glob
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report
np.random.seed(1337)
pd.options.display.width = 0
start_time = time.time()

#######################################################################################################################

# Setting 'Styles' Variable for use in plotting

styles = [':', '-.', '--', '-', ':', '-.', '--', '-', ':', '-.', '--', '-']

#######################################################################################################################

# Importing the data

dir_path = 'Data'
image_dir_path = os.path.join(dir_path, 'HAM10000')
print(image_dir_path)

image_id_path = {os.path.splitext(os.path.basename(x))[0]: x
                 for x in glob(os.path.join(image_dir_path, '*', '*.jpg'))}

# Creating a more human readable dictionary of cell type ID's

lesion_class_dict = {
    'nv': 'Melanocytic Nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign Keratosis-Like Lesions ',
    'bcc': 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratoses',
    'vasc': 'Vascular Lesions',
    'df': 'Dermatofibroma'
}
#######################################################################################################################

# 'dx' is short for diagnosis (for the patient) and "dx_type" is how the diagnosis was made.
# A bit more about each type of diagnosis and how they were made is available in the original paper:
# https://arxiv.org/abs/1803.10417

# Reading in the metadata file
ham_df = pd.read_csv(os.path.join(image_dir_path, 'HAM10000_metadata.csv'))

# Adding a Path to the corresponding image
ham_df['Path'] = ham_df['image_id'].map(image_id_path.get)

# Reformatting the diagnosis column to be increase readability
ham_df['Cell_Type'] = ham_df['dx'].map(lesion_class_dict.get)

# Creating a categorical Column for cell_type
ham_df['Cell_Type_id_Cat'] = pd.Categorical(ham_df['Cell_Type']).codes

# Checking for null values
print(ham_df.isnull().sum())

# Exploring the data frame

print('\n *****Exploring the data frame  \n')
print('\n ***DataFrame Info \n', ham_df.info())
print('\n ***DataFrame Nulls \n', ham_df.isnull().sum())

# Replace missing values using the mean along the axis with SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
age_impute = imputer.fit(ham_df[['age']])
ham_df['age'] = age_impute.transform(ham_df[['age']]).ravel()


#######################################################################################################################

# Function to print out frequencies of relevant Columns

def value_count(i):
    print('\n***', i, '***\n', ham_df[i].value_counts())


# Function to make a graph of frequencies for relevant Columns

def bar_chart(j):
    ham_df[j].value_counts().plot(kind='bar')
    plt.ylabel('Frequency')
    plt.xlabel(j)
    plt.tight_layout()
    plt.savefig("Results/HAM10000/Bar_Chart_" + j)
    plt.show()


cols = ham_df.columns.values.tolist()
wanted_cols = cols[2:7] + cols[8:]
print(wanted_cols)

for i in wanted_cols:
    value_count(i)
    bar_chart(i)

#######################################################################################################################

# Creating a column to hold the Np array for each image and resizing to 100 x 75

ham_df['Image'] = ham_df['Path'].map(lambda x: np.asarray(Image.open(x).resize((100, 75))))

# Checking that all Images have been resized and sent to arrays.

print(ham_df['Image'].map(lambda x: x.shape).value_counts())

print(ham_df.head())

#######################################################################################################################

# Printing 5 examples of each Cell Type

n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize=(4 * n_samples, 3 * 7))
for n_axs, (type_name, type_rows) in zip(m_axs,
                                         ham_df.sort_values(['Cell_Type']).groupby('Cell_Type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['Image'])
        c_ax.axis('off')
fig.savefig('Results/HAM10000/category_samples.png', dpi=300)

#######################################################################################################################

# Setting the Features and Target Variables

features = ham_df['Image']
target = ham_df['Cell_Type_id_Cat']

print(features.head())
print(target.head())

#######################################################################################################################

# Splitting the data into train, test and validation sets, using 73.5%, 10% and 16.5% of the data set respectively

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=101)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.165, random_state=101)

print("Shape of dataset: {}".format(str(features.shape)))
print("Shape of X_train: {}".format(str(X_train.shape)))
print("Shape of X_test: {}".format(str(X_test.shape)))
print("Shape of X_val: {}".format(str(X_val.shape)))

# Calculating the mean, Standard Deviation and Normalizing the X Train, Test and Val datasets

sets = ["train", "test", "val"]
for x in sets:
    globals()["X_{}".format(x)] = np.asarray(globals()["X_{}".format(x)].tolist())
    globals()["X_{}_mean".format(x)] = np.mean(globals()["X_{}".format(x)])
    globals()["X_{}_std".format(x)] = np.std(globals()["X_{}".format(x)])
    globals()["X_{}".format(x)] = (globals()["X_{}".format(x)] - globals()["X_{}_mean".format(x)]) / globals()[
        "X_{}_std".format(x)]

# Performing One-hot Encoding on the Targets

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)
y_val = to_categorical(y_val, num_classes=7)

# Reshaping the image arrays into 3 dimensions

X_train = X_train.reshape(X_train.shape[0], *(75, 100, 3))
X_test = X_test.reshape(X_test.shape[0], *(75, 100, 3))
X_val = X_val.reshape(X_val.shape[0], *(75, 100, 3))

#######################################################################################################################

# Applying data augmentation to Prevent Over fitting

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by standard deviation of the dataset
    samplewise_std_normalization=False,  # divide each input by its standard deviation
    zoom_range=0.1,  # Randomly zoom image
    rotation_range=10,  # randomly rotate images in the range 0 to 180 in 10 degrees steps
    width_shift_range=0.1,  # randomly shift images horizontally by a fraction of total height
    zca_whitening=False,  # apply ZCA whitening
    height_shift_range=0.1,  # randomly shift images vertically by a fraction of total height
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

#######################################################################################################################

# Defining Learning Rate Reduction, if no improvement on Val_Accuracy is seen over 3 epochs, the learning rate is halved

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

#######################################################################################################################

# Building CNN Model

# Declaring Variables

nets = 3
input_shape = (75, 100, 3)
num_classes = 7
model = [0] * nets

#######################################################################################################################

# *** Model 1 ***
# Running 3 separate models with varying numbers of convolutional subsampling pairs.
# Trying to get high accuracy and reduce Learning time.

for i in range(nets):
    model[i] = Sequential()
    model[i].add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model[i].add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model[i].add(MaxPool2D(pool_size=(2, 2)))
    if i > 0:
        model[i].add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
        model[i].add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
        model[i].add(MaxPool2D(pool_size=(2, 2)))
    if i > 1:
        model[i].add(
            Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
        model[i].add(
            Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
        model[i].add(MaxPool2D(pool_size=(2, 2)))
    model[i].add(Flatten())
    model[i].add(Dense(256, activation='relu'))
    model[i].add(Dense(num_classes, activation='softmax'))
    model[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = [0] * nets
names = ['Model 1.1', 'Model 1.2', 'Model 1.3']
epochs = 20
batch_size = 50
for i in range(nets):
    history[i] = model[i].fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                        epochs=epochs, validation_data=(X_val, y_val),
                                        verbose=0, steps_per_epoch=X_train.shape[0] // batch_size,
                                        callbacks=[learning_rate_reduction])
    print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(names[i],
                                                                                              epochs, max(
            history[i].history['accuracy']), max(history[i].history['val_accuracy'])))


# Plotting the Model Performance

plt.figure(figsize=(15, 5))
for i in range(nets):
    plt.plot(history[i].history['val_accuracy'], linestyle=styles[i])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(names, loc='upper left')
axes = plt.gca()
axes.set_ylim([0.72, 0.79])
plt.savefig('Results/HAM10000/Model_1_Accuracy')
plt.show()

#######################################################################################################################

# *** Model 2 ***
# Tuning the Number of Feature Maps in the Model
# Model 1.3 performed the strongest, with the highest Val set Accuracy and so will be used in this next step.

nets = 3
model = [0] * nets
for i in zip(range(nets), [32, 64, 128]):
    model[i[0]] = Sequential()
    model[i[0]].add(
        Conv2D(filters=i[1], kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model[i[0]].add(
        Conv2D(filters=i[1], kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model[i[0]].add(MaxPool2D(pool_size=(2, 2)))
    model[i[0]].add(
        Conv2D(filters=i[1] * 2, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model[i[0]].add(
        Conv2D(filters=i[1] * 2, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model[i[0]].add(MaxPool2D(pool_size=(2, 2)))
    model[i[0]].add(
        Conv2D(filters=i[1] * 4, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model[i[0]].add(
        Conv2D(filters=i[1] * 4, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model[i[0]].add(MaxPool2D(pool_size=(2, 2)))
    model[i[0]].add(Flatten())
    model[i[0]].add(Dense(256, activation='relu'))
    model[i[0]].add(Dense(num_classes, activation='softmax'))
    model[i[0]].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = [0] * nets
names = ['Model 2.1', 'Model 2.2', 'Model 2.3']
epochs = 20
batch_size = 50
for i in range(nets):
    history[i] = model[i].fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                        epochs=epochs, validation_data=(X_val, y_val),
                                        verbose=0, steps_per_epoch=X_train.shape[0] // batch_size,
                                        callbacks=[learning_rate_reduction])
    print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(names[i],
                                                                                              epochs, max(
            history[i].history['accuracy']), max(history[i].history['val_accuracy'])))


# Plotting the Model Performance

plt.figure(figsize=(15, 5))
for i in range(nets):
    plt.plot(history[i].history['val_accuracy'], linestyle=styles[i])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(names, loc='upper left')
axes = plt.gca()
axes.set_ylim([0.73, .82])
plt.savefig('Results/HAM10000/Model_2_Accuracy')
plt.show()

#######################################################################################################################

# *** Model 3 ***
# Model 2.2 is the strongest performer and will be used for the next stage
# Here Model 2.2 will be trained with 512 and 1024 Nodes in the Dense layer
# To investigate is there is a significant performance difference

nets = 2
model = [0] * nets

for i in range(2):
    model[i] = Sequential()
    model[i].add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model[i].add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model[i].add(MaxPool2D(pool_size=(2, 2)))
    model[i].add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model[i].add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model[i].add(MaxPool2D(pool_size=(2, 2)))
    model[i].add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model[i].add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model[i].add(MaxPool2D(pool_size=(2, 2)))
    model[i].add(Flatten())
    if i == 0:
        model[i].add(Dense(512, activation='relu'))
    elif i == 1:
        model[i].add(Dense(1024, activation='relu'))
    model[i].add(Dense(num_classes, activation='softmax'))
    model[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = [0] * nets
names = ['512N', '1024N']
epochs = 20
batch_size = 50
for i in range(nets):
    history[i] = model[i].fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                        epochs=epochs, validation_data=(X_val, y_val),
                                        verbose=0, steps_per_epoch=X_train.shape[0] // batch_size,
                                        callbacks=[learning_rate_reduction])
    print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(names[i],
                                                                                              epochs, max(
            history[i].history['accuracy']), max(history[i].history['val_accuracy'])))


# Plotting the Model Performance

nets = 2
names = ['512N', '1024N']

plt.figure(figsize=(15, 5))
for i in range(nets):
    plt.plot(history[i].history['val_accuracy'], linestyle=styles[i])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(names, loc='upper left')
axes = plt.gca()
axes.set_ylim([0.72, .82])
plt.savefig('Results/HAM10000/Model_3_Accuracy')
plt.show()

#######################################################################################################################

# Adding Batch Normalisation
# Batch Normalization is added between each Convolution Layer and Dense Layer to help reduce over fitting and
# to speed up convergence.
# Number of Epochs has also been increased
# Early Stopping has also been Introduced

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=6)
model_checkpoint = ModelCheckpoint(filepath='Final_Model.h5', monitor='val_loss', save_best_only=True)

history = [0]
epochs = 1
batch_size = 50

history[0] = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                 epochs=epochs, validation_data=(X_val, y_val),
                                 verbose=1, steps_per_epoch=X_train.shape[0] // batch_size,
                                 callbacks=[learning_rate_reduction, early_stop, model_checkpoint])

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
loss_v, accuracy_v = model.evaluate(X_val, y_val, verbose=1)
print("Test Set Accuracy = %f  ;  loss = %f" % (accuracy, loss))
print("Validation Set Accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

plt.figure(figsize=(15, 5))

plt.plot(history[0].history['val_accuracy'], linestyle=styles[0])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
axes = plt.gca()
axes.set_ylim([0.70, .85])
plt.savefig('Results/HAM10000/Final_Model_Accuracy')
plt.show()

#######################################################################################################################
print("--- %s seconds ---" % (time.time() - start_time))
#######################################################################################################################
