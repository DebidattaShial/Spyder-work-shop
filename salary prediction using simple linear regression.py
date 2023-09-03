import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'D:\machine learning\linear regression\1st\SIMPLE LINEAR REGRESSION\Salary_Data.csv')
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
from sklearn.linear_model import LinearRegression
regressoor=LinearRegression()
regressoor.fit(x_train, y_train)
y_pred=regressoor.predict(x_test)
plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, regressoor.predict(x_train),color="blue")
plt.title('Salary vs experiance(training set)')
plt.ylabel("Salary")
plt.show()
plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, regressoor.predict(x_train),color="blue")
plt.title('Salary vs experiance(testing set)')
plt.xlabel("year of experiance")
plt.ylabel("Salary")
plt.show()

m=regressoor.coef_

c=regressoor.intercept_
y_12 = 9312 * 12 + 26780
y_20 = 9312 * 20 + 26780
bias = regressoor.score(x_train, y_train)
bias
variance = regressoor.score(x_test, y_test)
variance