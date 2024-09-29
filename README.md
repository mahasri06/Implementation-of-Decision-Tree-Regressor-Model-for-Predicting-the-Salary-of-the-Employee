# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### Step 1: Start 
#### Step 2: Import the required libraries.
#### Step 3: Upload the csv file and read the dataset.
#### Step 4: Check for any null values using the isnull() function.
#### Step 5: From sklearn.tree import DecisionTreeRegressor.
#### Step 6: Import metrics and calculate the Mean squared error.
#### Step 7: Apply metrics to the dataset, and predict the output.
#### Step 8: End

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Mahasri P
RegisterNumber: 212223100029 
*/
```
```
import pandas as pd
data = pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position","Level"]]
y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =
train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

r2 = metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:
### Data Head
![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138849316/ce69b6cc-27ff-43fc-b430-22f987a7c65e)

### Data Info
![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138849316/3efcd4cd-17e8-4dfa-b6d4-fd6c1298c7ee)

### Data Head after applying LabelEncoder()
![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138849316/104c365e-0749-4cbb-a9d6-a79ed4d5d7b7)

### MSE
![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138849316/c73df1fd-2c70-47a3-8252-e226d47437b4)

### r2

![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138849316/ac848d8c-3dba-4403-ae2b-a2d98c29a44f)


### Data Prediction
![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138849316/6778bf4a-02b8-498b-b3aa-393f3a7e2b2f)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
