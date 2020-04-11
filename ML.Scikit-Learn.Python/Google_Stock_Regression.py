import pandas # L2
import quandl # L2
import math # L3
import numpy as np # L4
from sklearn import preprocessing, model_selection, svm  # L4
from sklearn.linear_model import LinearRegression # L4

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


# Lesson 4 - Regression training and testing.

X = np.array(training_data.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(training_data['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Linear regression
classifier = LinearRegression(n_jobs=-1)
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)

#  Support vector machine
SVM_classifier = svm.SVR(kernel='linear') # It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
SVM_classifier.fit(X_train, y_train)
SVMAccuracy = SVM_classifier.score(X_test, y_test)

print(accuracy)
print(SVMAccuracy)


# How to save to file with format csv ?

# ds = quandl.get('WIKI/GOOGL') 
# compression_opts = dict(method='zip', archive_name='out.csv')  
# ds.to_csv('out.zip', index=False, compression=compression_opts)