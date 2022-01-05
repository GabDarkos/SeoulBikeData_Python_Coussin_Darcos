# Final Projet of Python for Data Analysis
by Artémis Coussin and Gabriel Darcos

## Description of the problem

We had to analyze the Seoul Bike Sharing Demand Data Set using data visualization to show the links between the variables. 
We also needed to predict the Rented Bike count (Count of bikes rented at each hour) using the parameters we found relevant.
In order to do so, we built various models and kept the one giving us the best accuracy on the predictions on the test set (the minimum MSE).
We found the data on this [link](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand).

## What you will find in this repository

SeoulBikeData.csv : the dataset we worked on.  
Report_Coussin_Darcos.pptx : a power point explaining our project in every detail.  
Project_Python_Coussin_Darcos.ipynb : our code in Jupyter, with our graphs and models.  
model.pkl : the best model for our data prediction.  
app.py : our Flask application to predict a value of RentedBikeCount with the parameters you entered.  
index.html : the template we used for our Flask application.  


# 1. Data vizualisation

## Environment and tools

Here are the libraries required to launch our code and see every graph we made. 

import pandas as pd  
import csv  
import matplotlib.pyplot as plt  
import seaborn as sn  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn import metrics  
from sklearn.model_selection import GridSearchCV  
from sklearn.linear_model import Lasso  
from sklearn import linear_model  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.datasets import make_regression  
import plotly.express as px  
import xgboost as xgb  
import pickle  

For the installation, the line you should use is the next one : 
`pip install [name of the library]`  
OR  
`!pip install [name of the library]` if you do it directly on Jupyter

## The graphs

We did two types of graphics : 
1. Histograms on one variable to see its distribution.
2. Graphics between multiple variables to see how they interact with each other (we used mainly the library plotly).

You can see every graph in our Jupyter file : Projet_Python_Coussin_Darcos.ipynb


# 2. Predict RentedBikeCount

We had to predict how much bikes will be rented in a particular situation (hour, temperature, holiday, etc...)

RentedBikeCount is a continuous variable.
Hence, we will use models which predict a value and not a label : linear regression, lasso, decision tree, random forest and XGboost.  

To find the best values for the parameters of a model, we did a grid search.

We measured the accuracy of each model using the MSE (Mean Squared Error). 

## Linear Regression

We found a MSE of 104,77.

## Lasso

We used these values on the parameters :  
alpha = 7   
max_iter = 100000  

We found a MSE of 104,85.

## Decision Tree

We used these values on the parameters :  
criterion = 'friedman_mse'  
max_depth = 7  
splitter = 'best'  

We found a MSE of 63.737.

## RandomForest

We used these values on the parameters :  
max_depth = 18  
n_estimators = 300  

We found a MSE of 46.945.

## XGBoost

We used these values on the parameters :  
max_depth = 7  
learning_rate = 0.1  
colsample_bytree = 0.9  
alpha = 12  
n_estimators = 200. 

We found a MSE of 45.044.

## Choice of the model

The model with the best MSE was XGBoost. However, we encountered some problems of conversion that forced us to put validation_features = false, which was risky because you could
enter any values and it would not be checked. 
Plus, there was a difference of only 1 between the best and second best MSEs. 

Hence, we chose the RandomForest model with a MSE of 46.945. 


# 3. Flask

The pickle module (standard in Python) allows the serialization of memory objects into strings (and vice versa). 
It is useful for persistence and data transfer over a network. We used it to save our RandomForest model from our Jupyter code : it's the file called model.pkl

We created a html file for the structure of the browser window of our Flask, it is called index.html and can be found in our Template file.

Finally, our Flask code is called app.py and **it must be ran before clicking on the link of our browser window**. 

You can find our Flask at this link : http://127.0.0.1:5000/


Thanks for your reading, we hope you will enjoy our project !
Artémis and Gabriel
