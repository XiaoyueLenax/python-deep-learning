from google.colab import drive
import os as os
from requests import get

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

from tensorflow import keras
#from tensorflow.keras import layers
from keras import layers
#from tensorflow.keras import losses
from keras import losses

# Cannot load these two modules at home. Ask in class or try again
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor

import sklearn
sklearn.__version__  #1.3.0

def download_save(url, filename):
 # Here, it says get is not defined. **
  res = get(url)
  if res.status_code != 200:
    print(f"Couldn't fetch data from {url}")
  else:
    csv_file = open(filename, 'wb')
    csv_file.write(res.content)
    csv_file.close()
    
download_save('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
              'wine.csv')

df_wine = pd.read_csv('wine.csv', sep=';')
print(df_wine.shape)
df_wine.sample(10)

download_save('https://gist.githubusercontent.com/ktisha/c21e73a1bd1700294ef790c56c8aec1f/raw/819b69b5736821ccee93d05b51de0510bea00294/pima-indians-diabetes.csv',
              'pima.csv')

df_pima = pd.read_csv('pima.csv',
                      header=8,
                      names = ['preg', 'gluc', 'pres', 'skin' ,'insu', 'bmi', 'pedi', 'age', 'class'])
df_pima.sample(10)

#------- Regression---------
X = df_wine.copy()
y = X.pop('quality')
scaler = MinMaxScaler((-1, 1))

#------- Standardization ------
X = df_wine.copy()
y = X.pop('quality')

scaler = MinMaxScaler((-1, 1))

X = pd.DataFrame(scaler.fit_transform(X),
                       columns=X.columns,
                       index=X.index)

# ---------- Linear Regression -------------
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)
mse = mean_squared_error(y, predictions)
print(mse)

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(y=X.columns, x=model.coef_, orient='h')
plt.show()


# ----------- Build -----------------
input_shape = X.shape[1]

model = keras.Sequential([
    layers.Dense(units=1,
                 input_shape=[input_shape],
                 activation=None)
])
model.summary()

# ------ Compile ------------
model.compile(loss=losses.MeanSquaredError())

# --------- Train -----------
%%time
history = model.fit(X,
                    y,
                    epochs=1000,
                    verbose=0)

#----------- Predict ---------------
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)
print(mse)

model.weights[0].numpy().flatten()

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(y=X.columns,
            x=model.weights[0].numpy().flatten(),
            orient='h')
plt.show()

# -------- Pipelines ---------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)


# -------- Linear Regression -----------
model = Pipeline([('scaler', MinMaxScaler((-1, 1))),
                  ('regressor', LinearRegression())])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(mse)

predictions = model.predict(X_train)
mse = mean_squared_error(y_train, predictions)
print(mse)

# ----- Perceptron -----------
def create_perceptron(input_shape):
  model = keras.Sequential([
    layers.Dense(units=1,
                 input_shape=[input_shape],
                 activation=None)
  ])

  model.compile(loss=losses.MeanSquaredError())

  return model

perceptron = KerasRegressor(build_fn=create_perceptron,
                            input_shape=X_train.shape[1],
                            epochs=1000,
                            verbose=0)

model = Pipeline([('scaler', MinMaxScaler((-1, 1))),
                  ('regressor', perceptron)])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(mse)

predictions = model.predict(X_train)
mse = mean_squared_error(y_train, predictions)
print(mse)

#----------------------------- The new way -----------------------
# Migration to how to do it from now on https://www.adriangb.com/scikeras/stable/migration.html
!pip install -qq scikeras


from scikeras.wrappers import KerasClassifier, KerasRegressor

perceptron = KerasRegressor(#build_fn=create_perceptron,
                            model=create_perceptron,
                            input_shape=X_train.shape[1],
                            epochs=1000,
                            verbose=0)

model = Pipeline([('scaler', MinMaxScaler((-1, 1))),
                  ('regressor', perceptron)])

history = model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(mse)

#-------- Classification --------------
df_pima.sample(5)
X = df_pima.copy()
y = X.pop('class')

# -------- Logistic Regression --------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

model = Pipeline([('scaler', MinMaxScaler((-1, 1))),
                  ('classifier',  LogisticRegression(max_iter=1000))])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(acc)

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(y=[*X.columns,
               'intercept'],
            x=[*model.named_steps['classifier'].coef_[0],
               model.named_steps['classifier'].intercept_[0]],
            orient='h')
plt.show()

# -------- Perceptron ------------
def create_perceptron(input_shape):
  model = keras.Sequential([
    layers.Dense(units=1,
                 input_shape=[input_shape],
                 activation='sigmoid')
  ])

  model.compile(loss=losses.MeanSquaredError())

  return model

perceptron = KerasClassifier(build_fn=create_perceptron,
                             input_shape=X_train.shape[1],
                             epochs=1000,
                             verbose=0)

model = Pipeline([('scaler', MinMaxScaler((-1, 1))),
                  ('classifier', perceptron)])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(acc)

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(y=[*X.columns,
               'intercept'],
            x=[*model.named_steps['classifier'].model.weights[0].numpy().flatten(),
               *model.named_steps['classifier'].model.weights[1].numpy().flatten()],
            orient='h')
plt.show()

#---------- Exercises ------------------ **