# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values
## Program:
```py

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sarankumar J
RegisterNumber: 212221230087

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![image](https://user-images.githubusercontent.com/94778101/201474359-dbcfdc05-49f1-4eed-84c6-d4eb6033dc21.png)

![image](https://user-images.githubusercontent.com/94778101/201474370-8e631187-090d-4f12-bf44-b2b4becf9024.png)

![image](https://user-images.githubusercontent.com/94778101/201474387-815a5e19-d508-4044-ab41-1bef946c6721.png)

![image](https://user-images.githubusercontent.com/94778101/201474398-f2487248-b9ee-410b-989c-7c3ff2b820b8.png)

![image](https://user-images.githubusercontent.com/94778101/201474418-75875ffe-0c27-455f-8088-6e1af07f5670.png)

![image](https://user-images.githubusercontent.com/94778101/201474443-822d2e49-7b49-45d8-8f88-8e3f7790e3ac.png)

![image](https://user-images.githubusercontent.com/94778101/201474457-f15ee82a-ec7a-4f02-be72-0e7b28c06220.png)

![image](https://user-images.githubusercontent.com/94778101/201474468-b7bb1118-d7db-4b94-8213-542073979370.png)

![image](https://user-images.githubusercontent.com/94778101/201474499-c087cf39-8508-4f02-b8f8-4078b8588d8b.png)

![image](https://user-images.githubusercontent.com/94778101/201474546-c4c8b8af-4cdb-4d9b-a7ad-8307a0b6ad94.png)

![image](https://user-images.githubusercontent.com/94778101/201474560-aa344111-48dd-44e6-b1b7-86cd404cf7ac.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
