import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def encoding(data):

    encoder_type=input('1 OneHotEncoder,2 LabelEncoder')
    if int(encoder_type)==1:
        encode_num=input('Column Number For Encode:')
        print("Encoding...")
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [int(encode_num)])], remainder='passthrough')
        data = np.array(ct.fit_transform(data))
    elif int(encoder_type)==2:
        print("Warning: All data will be encoded with Label encoder...")
        print("Encoding...")
        le = LabelEncoder()
        data = le.fit_transform(data)

    return data


while True:
 
    dataset_name=input("Dataset Name:")
    dataset = pd.read_csv(str(dataset_name))
    dependent_variable=input("The dependent variable:")
    X=dataset.drop(str(dependent_variable),axis=1)
    y=dataset[str(dependent_variable)]
    choise=input('Do you want encoding(y,n) :')
    if str(choise)=="y":
      encoding_variable=input("Encoding Variable (x or y) :")
      if str(encoding_variable)=="x":
          X=encoding(X)
      elif str(encoding_variable)=="y":
          y=encoding(y)


    test_size=input("Test Size(ex: 0.2) :")
    print("Splitting train and test data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = float(test_size), random_state = 42)

    print("Regression model is training...")
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    print("Predicting results (X_test data)...")
    y_pred = regressor.predict(X_test)

    print("Printing results...")
    print("Results: "+str(y_pred))

    print("Comparison with test data...")
    y_pred=pd.Series(y_pred)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.values.reshape(len(y_pred),1), y_test.values.reshape(len(y_test),1)),1))

    conchoise=input("Do you want to exit (y or n):")

    if str(conchoise)=="y":
        break
