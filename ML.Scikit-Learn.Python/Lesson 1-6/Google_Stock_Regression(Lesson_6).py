import pandas # L2
import quandl # L2
import math # L3
import numpy as np # L4
from sklearn import preprocessing, model_selection, svm  # L4
from sklearn.linear_model import LinearRegression # L4
from datetime import datetime # L5
import matplotlib.pyplot as plot # L5
from matplotlib import style  # L5 
import pickle # L6

style.use('ggplot')

# Lesson 2 - Read data google stock from quandl. First iteration of training data preparation.
raw_data = pandas.read_csv('out.zip')

selective_data = raw_data[[ 'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

selective_data['HL_PCT'] = (selective_data['Adj. High'] - selective_data['Adj. Close']) / selective_data['Adj. Close'] * 100.0
selective_data['PCT_change'] = (selective_data['Adj. Close'] - selective_data['Adj. Open']) / selective_data['Adj. Open'] * 100.0

training_data = selective_data[[ 'Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume' ]]


# Lesson 3 - Replace Nan data to -9999. Create label and forecast out.
forecast_col = 'Adj. Close'
training_data.fillna(-9999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(training_data)))

training_data['label'] = training_data[forecast_col].shift(-forecast_out)
training_data.dropna(inplace=True)


# Lesson 3-4 - Regression training and testing.

training_data.dropna(inplace=True)
X = np.array(training_data.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

training_data.dropna(inplace=True)
y = np.array(training_data['label'])
y_lately = y[-forecast_out:]
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Linear regression
classifier = LinearRegression(n_jobs=-1)
classifier.fit(X_train, y_train)

# Save linear regression model to file.
with open('linear_regression.pickle', 'wb') as file:
    pickle.dump(classifier, file)

# Read model.
pickle_in = open('linear_regression.pickle', 'rb')
classifier = pickle.load(pickle_in)

accuracy = classifier.score(X_test, y_test)

#  Support vector machine
SVM_classifier = svm.SVR(kernel='linear') # It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
SVM_classifier.fit(X_train, y_train)
SVMAccuracy = SVM_classifier.score(X_test, y_test)

print(accuracy)
print(SVMAccuracy)

forecast_set = classifier.predict(X_lately)
print(forecast_set)

# Prediction future price. (own code) 
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
print(y_lately - forecast_set)


training_data['Forecast'] = np.nan

last_date = training_data.iloc[-1].name
last_unix = pandas.to_datetime(last_date).timestamp()
one_day = 86400
next_unit = last_unix + one_day

for i in forecast_set:
    next_date = datetime.fromtimestamp(next_unit)
    next_unit += one_day
    training_data.loc[next_date] = [np.nan for _ in range(len(training_data.columns) - 1)] + [i]


training_data['Adj. Close'].plot()
training_data['Forecast'].plot()
plot.legend(loc=4)
plot.xlabel('Date')
plot.xlabel('Price')
plot.show()


