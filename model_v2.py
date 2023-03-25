import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

student_dataset = pd.read_csv('dataa.csv')
print("\nprinting the first 5 rows of the dataset\n")
print(student_dataset.head())

print("\nRows & Colums = "+str(student_dataset.shape))
print(student_dataset.describe())

print(student_dataset['Dropout'].value_counts())
print(student_dataset.groupby('Dropout').mean())

print("separating the data and labels")
X = student_dataset.drop(columns = 'Dropout', axis=1)
Y = student_dataset['Dropout']


print(X)
print("\n\n")
print(Y)
print("\n\n")

print("Data Standardization")
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
print("\n\n")

print("Training the Model")
classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)


input_data = (1,1,1,1,1,1,1,1,1,1,0)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)

prediction = classifier.predict(std_data)
print(prediction)


if (prediction[0] == 0):
  print('The student will not dropout')
else:
  print('The student will dropout')
