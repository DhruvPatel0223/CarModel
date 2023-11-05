import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("imports-85.data")
print(data.head())

data = data[["horsepower", "city-mpg", "highway-mpg", "price"]]

predict = "price"

x = np.array(data.drop(predict, axis = 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)
best = 0
'''
for _ in range(50):
    #The model learns from x_train and uses y_train as the corresponding target values to adjust its parameters and make predictions
    #After the model has been trained on x_train and y_train, it is tested on x_test, and its predictions are compared to the actual target values in y_test to evaluate its performance.
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)

    #linear regression model
    linear = linear_model.LinearRegression()
    #creates best-fit line using x_train and y_train
    linear.fit(x_train, y_train)
    #tests accuracy of model
    acc = linear.score(x_test, y_test)
    print(acc)
#only going to save a new model if the current accuracy is better than any previous one we've seen
    if acc > best:
        best = acc
        with open("carmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
            '''
'''
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)
'''


pickle_in = open("carmodel.pickle", "rb")
#loads model in the variable called linear
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
    