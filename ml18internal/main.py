import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
import pickle

urldata = pd.read_csv("Url_Processed.csv")

# droping "Unnamed: 0" as its unncessary feature
urldata.drop("Unnamed: 0",axis=1,inplace=True)

urldata.head(10)
urldata.drop(["url","label"],axis=1,inplace=True)

# print(urldata.head())
#Independent Variables
x = urldata[['hostname_length',
       'path_length', 'fd_length', 'count-', 'count@', 'count?',
       'count%', 'count.', 'count=', 'count-http','count-https', 'count-www', 'count-digits',
       'count-letters', 'count_dir', 'use_of_ip']]

#Dependent Variable
y = urldata['result']
#Oversampling using SMOTE
print("helo")
# sample=SMOTE()
# x_sample, y_sample = sample.fit_resample(x, y.values.ravel())
# print("smore")
# x_sample = pd.DataFrame(x_sample)
# y_sample = pd.DataFrame(y_sample)
# # checking the sizes of the sample data
# print("Size of x-sample :", x_sample.shape)
# print("Size of y-sample :", y_sample.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
print("Shape of x_train: ", x_train.shape)
print("Shape of x_valid: ", x_test.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y_valid: ", y_test.shape)


model = DecisionTreeClassifier(max_depth=7)


# start training the model
model.fit(x_train.values,y_train)
predictions = model.predict([
    [0, 25, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 22, 0, 1]
])
print(predictions)
# from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

# # finding y_pred for train and test dataset
# y_pred_train = model.predict(x_train.values)
# y_pred_test = model.predict(x_test)

# # finding accuracy on train dataset
# train_acc = accuracy_score(y_train,y_pred_train)
# test_acc = accuracy_score(y_test,y_pred_test)

# print("Accuracy on Training dataset : ",round(train_acc,3))
# print("Accuracy on Testing dataset : ",round(test_acc,3))
# filename='url.pkl'
# pickle.dump(model,open(filename,'wb'))
