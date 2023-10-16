import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from keras.layers import Dense,LSTM
from keras.callbacks import EarlyStopping,ModelCheckpoint

store_sales = pd.read_csv("C:\\Git\\Project\\train.csv")
store_sales=store_sales.drop(["store","item"],axis=1)
store_sales['date']=pd.to_datetime(store_sales['date'])
store_sales["date"] = store_sales["date"].dt.to_period("M")
monthly_sales = store_sales.groupby('date').sum().reset_index()
monthly_sales['date']=monthly_sales['date'].dt.to_timestamp()
plt.figure(figsize=(15,5))
plt.plot(monthly_sales['date'],monthly_sales['sales'])
plt.xlabel("date")
plt.ylabel("sales")
plt.title("monthly customer sales")
plt.show()
monthly_sales['sales_diff']=monthly_sales['sales'].diff()
monthly_sales = monthly_sales.dropna()
print(monthly_sales.head(10))
#print(store_sales.head(10))
