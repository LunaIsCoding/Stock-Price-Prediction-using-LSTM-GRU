import streamlit as st
from datetime import date
# import yfinance as yf
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.layers import *
from prophet import Prophet
from prophet.plot import plot_plotly, plot_yearly
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from math import sqrt
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

trainBtnClicked = False
st.title("Stock Price Prediction Web Application")

# stocks=("AaPL", "GOOG", "MSFT", "GME")
# selected_stocks = st.selectbox("select dataset for Prediction", stocks)
filepath = st.text_input("Dataset File Url", "AAPL.csv")
colName = st.text_input("Prediction Column Name", "Close")

# days = st.text_input("How many days you want to predict", "")
days = st.slider("Days to predict", 1, 60)

#Convert date
def to_strftime(df):
    date = datetime.strptime(df, '%Y-%m-%d')
    return date.strftime("%Y-%m-%d")

def import_data():
	st.subheader("Last 5 Rows of Dataset:")
	dataframe = pd.read_csv(filepath)
	# dataframe['Date'] = dataframe['Date'].apply(lambda x: to_strftime(x))
	# dataframe = dataframe.sort_values('Date').reset_index(drop=True)
	# dataframe = yf.download("AAPL", START, TODAY)
	# dataframe.reset_index(inplace=True)
	# dataframe.to_csv("AAPL.csv")
	# dataframe.drop([dataframe.columns[0]], axis=0, inplace=True)
	dataframe.drop(dataframe.columns[dataframe.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
	# dataframe = dataframe.sort_values('Date').reset_index(drop=True)
	
	st.write(dataframe.tail())
	return dataframe

def plot_data(df):
	figure = go.Figure()
	# figure.add_trace(go.Scatter(x=df['Date'], y = df['Open'], name="Predicted " + colName))
	figure.add_trace(go.Scatter(x=df['Date'], y = df[colName], name="Predicted " + colName))
	figure.layout.update(title_text="Date vs " + colName, xaxis_rangeslider_visible=True)
	st.plotly_chart(figure)

def plot_trends(df, x, y):
	print(df.columns[0])
	figure = go.Figure()
	# figure.add_trace(go.Scatter(x=df['Date'], y = df['Open'], name="Predicted " + colName))
	figure.add_trace(go.Scatter(x=df[df.columns[0]], y = df[df.columns[1]], name="Predicted"))
	figure.layout.update(title_text=x +" vs "+ y, xaxis_rangeslider_visible=True)
	st.plotly_chart(figure)

def prepare_data(df):
	df[colName] = df[colName].astype(float)
	num_shape = df.shape[0] - 500
	window = 60
	# st.write(df.shape)
	# train/test split
	train = df.iloc[:num_shape, 1:2].values
	test = df.iloc[num_shape:, 1:2].values
	df_price = np.vstack((train, test))

	sc = MinMaxScaler(feature_range = (0, 1))

	X_train, Y_train = [], []
	# print(type(train))
	# st.write(train)
	train_scaled = sc.fit_transform(train)

	for i in range(train.shape[0]-window):
	    batch = np.reshape(train_scaled[i:i+window, 0], (window, 1))
	    X_train.append(batch)
	    Y_train.append(train_scaled[i+window, 0])
	X_train = np.stack(X_train)
	Y_train = np.stack(Y_train)

	X_test, Y_test = [], []
	test_scaled = sc.fit_transform(df_price[num_shape-window:])

	for i in range(test.shape[0]):
	    batch = np.reshape(test_scaled[i:i+window, 0], (window, 1))
	    X_test.append(batch)
	    Y_test.append(test_scaled[i+window, 0])

	return X_train, Y_train, X_test, Y_test, sc

def LSTM_model_training(X_train, Y_train, X_test, Y_test, sc):
	LSTM_model = Sequential()
	LSTM_model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
	LSTM_model.add(Dropout(0.2))

	LSTM_model.add(LSTM(units = 50, return_sequences = True))
	LSTM_model.add(Dropout(0.2))

	LSTM_model.add(LSTM(units = 50, return_sequences = True))
	LSTM_model.add(Dropout(0.2))

	LSTM_model.add(LSTM(units = 50))
	LSTM_model.add(Dropout(0.2))

	LSTM_model.add(Dense(units = 1))

	LSTM_model.compile(optimizer = 'adam', loss = 'mse')
	LSTM_model.fit(X_train, Y_train, epochs = 100, batch_size = 32, verbose=1)

def GRU_model_training(X_train, Y_train, X_test,Y_test, sc):
	# GRU architecture
	GRU_model = Sequential()

	GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
	GRU_model.add(Dropout(0.2))

	GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
	GRU_model.add(Dropout(0.2))

	GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
	GRU_model.add(Dropout(0.2))

	GRU_model.add(GRU(units=50))
	GRU_model.add(Dropout(0.2))

	GRU_model.add(Dense(units=1))
	
	GRU_model.compile(optimizer='sgd', loss='mean_squared_error')
	GRU_model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1)

def train(df):
	dfTrain = df[['Date', colName]]
	dfTrain = dfTrain.rename(columns = {"Date": "ds", colName: "y"})
	X_train, Y_train, X_test, Y_test, sc = prepare_data(df)	
	LSTM_model_training(X_train, Y_train, X_test, Y_test, sc)
	GRU_model_training(X_train, Y_train, X_test, Y_test, sc)
	st.write("Training completed!")
	st.write("Please wait for a while, we are displaying output predictions for you & evaluating it...")
	m = Prophet(interval_width=0.95, daily_seasonality=True, yearly_seasonality=True)
	m.fit(dfTrain)
	p = m.make_future_dataframe(periods = days, freq='D')
	f = m.predict(p)
	pDf = f[['ds', 'yhat']]
	pDf = pDf.rename(columns = {"ds": "Date", "yhat": "Predicted Price"})
	st.write("R2 Score is: " + str(r2_score(list(df[colName]), list(pDf['Predicted Price'].iloc[:-int(days)]))))
	st.write("RMSE Score is: " + str(sqrt(mean_squared_error(list(df[colName]), list(pDf['Predicted Price'].iloc[:-int(days)])))))

	st.subheader("Predicted Data")
	st.write(pDf.tail())
	st.subheader("Prediction Plot")
	
	figure1 = plot_plotly(m, f)
	st.plotly_chart(figure1)

	# figure1 = m.plot(f, xlabel='Date', ylabel='Price')
	# ax = figure1.gca()
	# ax.set_title("Date vs Price", size=25)
	# ax.set_xlabel("Date", size=24)
	# ax.set_ylabel("Price", size=24)
	# ax.tick_params(axis="x", labelsize=14)
	# ax.tick_params(axis="y", labelsize=14)
	# st.write(figure1)

	st.subheader("Prediction Trends")
	figure2 = m.plot_components(f)
	figure2.get_children()[1].set_xlabel('Name of year')
	figure2.get_children()[1].set_ylabel('yearly')
	figure2.get_children()[2].set_ylabel('daily')
	figure2.get_children()[3].set_xlabel('month in year')
	figure2.get_children()[3].set_ylabel('monthly')
	figure2.get_children()[4].set_ylabel('hourly')
	st.write(figure2)

	print(f.columns)
	# plot_trends(f[['ds', 'trend']], "Name of Years", "Yearly")
	# plot_trends(f[['ds', 'yearly']], "Name of Months", "Monthly")
	# plot_trends(f[['ds', 'weekly']], "Name of days", "Daily")
	# plot_trends(f[['ds', 'daily']], "Time hourly", "Hourly")

	# figYearly = plot_yearly(m)
	# st.write(figYearly)

if st.button('Start Training'):
	DF = import_data()
	plot_data(DF)
	st.write("Model is started training. Please wait for few minutes...")
	train(DF)
	


print("Website Loaded!")
