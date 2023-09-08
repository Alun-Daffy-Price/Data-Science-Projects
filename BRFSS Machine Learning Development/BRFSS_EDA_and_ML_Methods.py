# Imports
import random
import time
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, plot_confusion_matrix, plot_roc_curve, \
    plot_precision_recall_curve, classification_report, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.simplefilter(action='ignore', category=FutureWarning)
start_time = time.time()
tf.random.set_seed(2022)
random.seed(1337)
pd.options.display.width = 0

#####################################################################################################################
# Importing Datasets

clean_df = pd.read_csv(f'Data/BRFSS/Cleaned_Dataset.csv')
balanced_df = pd.read_csv(f'Data/BRFSS/Balanced_Dataset.csv')

print(balanced_df.head())
print(clean_df.head())


# Data Exploration Function

def df_explore(df):
    print('*** Dataframe descriptive Statistics*** \n', df.describe())
    print('*** Dataframe Info *** \n', df.info())
    print('*** Dataframe Shape *** \n ', df.shape)


# Heatmap Correlation Function

def heatmap(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), vmax=0.9, square=True)
    plt.title("Pearson Correlation")
    plt.savefig("Results/BRFSS/Heat_Map")
    plt.show()


# Exploring Data

df_explore(clean_df)
df_explore(balanced_df)

heatmap(clean_df)
heatmap(balanced_df)

#####################################################################################################################
# Function to print frequencies of relevant Columns

cols = clean_df.columns.values.tolist()


def value_count(col):
    print('\n***', col, '***\n', clean_df[col].value_counts())


# Function to make a graph of frequencies for relevant Columns

def bar_chart(j):
    clean_df[j].value_counts().plot(kind='bar')
    plt.ylabel('Frequency')
    plt.xlabel(j)
    plt.tight_layout()
    plt.savefig("Results/BRFSS/Bar_Chart_" + j)
    plt.show()


for col in cols:
    value_count(col)
    bar_chart(col)

#####################################################################################################################
# Checking for class imbalance

print(clean_df['Diabetes'].value_counts())

# Separating the features and target

X = clean_df.drop('Diabetes', axis=1)
y = clean_df['Diabetes']

# Removing the class imbalance

under_sample = RandomUnderSampler(sampling_strategy=0.25)
X_under, y_under = under_sample.fit_resample(X, y)

print(X_under.shape)
print(y_under.shape)
print(y_under.value_counts())

#####################################################################################################################
# testing NN on clean df, using random under sampling.

# Splitting the data into a train and test group using a 20%/80% split

X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size=0.20, random_state=101)

# Scaling the Data

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#####################################################################################################################
# Creating a Deep Neural Network


# Initialising the ANN
ANN_Model = Sequential()

# Adding the input layer and the first hidden layer
ANN_Model.add(Dense(6, activation='relu'))

# Adding the second hidden layer
ANN_Model.add(Dense(6, activation='relu'))

# Adding the third hidden layer
ANN_Model.add(Dense(6, activation='relu'))

# Adding the output layer
ANN_Model.add(Dense(1, activation='sigmoid'))

# Compiling the ANN
ANN_Model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Fit the ANN_Model on the training Data
history = ANN_Model.fit(X_train, y_train, epochs=100)

#####################################################################################################################
# Plotting Loss and Accuracy of the ANN_Model


pd.DataFrame(history.history).plot()
plt.ylabel('loss')
plt.xlabel('accuracy')
plt.title("Loss Accuracy Plot of ANN_Model ")
plt.savefig("Results/BRFSS/Loss_Accuracy_Plot_ANN_Model")
plt.show()

print(ANN_Model.evaluate(X_test, y_test))


#######################################################################################################################

p_pred = ANN_Model.predict(X_test)

p_pred = p_pred.flatten()
print(p_pred.round(2))

y_pred = np.where(p_pred > 0.5, 1, 0)
print(y_pred)

print(classification_report(y_test, y_pred))
class_report = classification_report(y_test, y_pred, output_dict=True)
class_report_df = pd.DataFrame.from_dict(class_report)
print(class_report)
class_report_df.to_csv("Results/BRFSS/Classification_Report_ANN_Model.csv")

#####################################################################################################################
# Creating an evaluation function

def model_metrics(model):
    y_pred_1 = model.predict(X_test)
    print("***Classification Report For: " + str(model) + "***")
    class_report = classification_report(y_test, y_pred_1, output_dict=True)
    class_report_df = pd.DataFrame.from_dict(class_report)
    print(class_report)
    class_report_df.to_csv("Results/BRFSS/Classification_Report_" + str(model) + ".csv")
    plot_confusion_matrix(model, X_test, y_test)
    plt.title("Confusion_Matrix_" + str(model))
    plt.savefig("Results/BRFSS/Confusion_Matrix_" + str(model))
    plt.show()
    plot_roc_curve(model, X_test, y_test)
    plt.title("ROC_Curve" + str(model))
    plt.savefig("Results/BRFSS/ROC_Curve" + str(model))
    plt.show()
    plot_precision_recall_curve(model, X_test, y_test)
    plt.title("Precision_Recall_Curve_" + str(model))
    plt.savefig("Results/BRFSS/Precision_Recall_Curve_" + str(model))
    plt.show()


def accuracies(model):
    model_accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10, n_jobs=-1)
    print(str(model) + " Accuracy Standard Deviation = ", model_accuracies.std())
    print(str(model) + " Accuracy Mean = ", model_accuracies.mean())


#####################################################################################################################
# Using a KNN Algorithm

# Determining the best K value to use

test_error = []
iterations = list(range(1, 31))
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)

    test_error.append(error)
    print(k, error)

# Plotting the error rate

plt.figure(figsize=(8, 4), dpi=150)
plt.plot(range(1, 31), test_error)
plt.savefig("Results/BRFSS/Optimum_K_Value_Graph")
plt.show()

# find lowest error rate

t_error = zip(iterations, test_error)
test_error_dict = dict(t_error)
optimum_k_value = min(test_error_dict, key=test_error_dict.get)
print(test_error_dict)
print("Optimum K Value = ", optimum_k_value)

# Running KNN with optimum K value

knn_model = KNeighborsClassifier(n_neighbors=optimum_k_value, metric='minkowski', p=2)
knn_model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = knn_model.predict(X_test)

# Evaluating the KNN Model

model_metrics(knn_model)

# applying k-fold cross validation

accuracies(knn_model)

#####################################################################################################################
# Running a Random Forest Classifier

random_forest_model = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0)
random_forest_model.fit(X_train, y_train)

# Predicting the Test set results
y_pred_RF = random_forest_model.predict(X_test)

# Evaluating the Model

model_metrics(random_forest_model)

# applying k-fold cross validation

accuracies(random_forest_model)

#####################################################################################################################
# Creating a Naive Bayes

# Training the Naive Bayes model on the Training set

naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)

# Predicting the Test set results
y_pred_nb = naive_bayes_model.predict(X_test)

# Evaluating the Model

model_metrics(naive_bayes_model)

# applying k-fold cross validation

accuracies(naive_bayes_model)

#####################################################################################################################
# Creating an SVM

# Training the SVM model on the Training set

svm_model = SVC(kernel='linear', random_state=0)
svm_model.fit(X_train, y_train)

# Predicting the Test set results
y_pred_svm = svm_model.predict(X_test)

# Evaluating the Model

model_metrics(svm_model)

# applying k-fold cross validation

accuracies(svm_model)

#####################################################################################################################
# Creating a Kernel SVM
#
# Training the Kernel SVM model on the Training set

kernel_svm = SVC(kernel='rbf', random_state=0)
kernel_svm.fit(X_train, y_train)

# Predicting the Test set results
y_pred_SVM_K = kernel_svm.predict(X_test)

# Evaluating the Model

model_metrics(kernel_svm)

# applying k-fold cross validation

accuracies(kernel_svm)

#####################################################################################################################
print("--- %s seconds ---" % (time.time() - start_time))
