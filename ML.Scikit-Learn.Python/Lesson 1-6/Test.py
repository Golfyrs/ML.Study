import pandas # L1
import quandl # L1
import math # L2

raw_data = pandas.read_csv('out.zip')

data = raw_data[:10]

print(data)

forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.3 * len(data)))

data['label'] = data[forecast_col].shift(-forecast_out)


print("forecast_out: " , forecast_out)
print(data)