# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:
![Screenshot 2024-02-23 113616](https://github.com/RamkumarGunasekaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870820/78dde68e-fffb-42d8-8c31-706abbb66489)
![Screenshot 2024-02-23 113801](https://github.com/RamkumarGunasekaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870820/3854d5f2-aa78-47b5-a13f-532c0571a8ca)
![Screenshot 2024-02-23 113838](https://github.com/RamkumarGunasekaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870820/923f8313-701f-4cca-8e23-6aa9653d56ec)
![Screenshot 2024-02-23 113905](https://github.com/RamkumarGunasekaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870820/38eeefe0-925e-4b93-b1a6-cfaa35fef2c2)
![Screenshot 2024-02-23 113933](https://github.com/RamkumarGunasekaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870820/6d5e0b6f-8cfb-4be9-8f8d-72d6dac96049)







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
