#Fetch Data From Yahoo Finance Website Using API.
#Forecasting Or Prediction Modules are Used *Facebook Prophet*
#Modules Use- Streamlit Web App Data and Time Module For Forecasting, yfinance(Yahoo Finance) module, fbprophet.plot module,
#plotly is used For plots genrates(High Api and fancy)

import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


Dt1="2016-01-01"       #Start Date
Dt2= date.today().strftime("%Y-%m-%d")  #Today- End Date.  strftime- Covert Into String Format

#Title-
st.title("Stock  Market Prediction App")
 
#Some Famous Company Mentioned Like Reliance,Tcs,Wipro,Infosys.
#N.s Means National Stocks Exchange , B.o Means Bombay Stock Exchange 
#Company Code- Reliance, Tcs, WIT, INFY
stocks= ("RELIANCE.NS","TCS.BO","TCS.NS","WIT","INFY")  

#Dropdown Menu-st.selectbox-
Selected_stock= st.selectbox("Choose Your Dataset For Prediction",stocks)
#st.slider- How Many Year of Prediction You Want And Next 5 Year How Much Stocks Increases.
N_years = st.slider("Choose Your Year of Prediction",1,5)

#Actual Year Covert Into N_years*365 ...
period=N_years* 365

#Data Loading 
data_load_state = st.text("Loading data...")

#yf.download- Selected_stock  From where do we need the data-Dt1 ,Dt2...
data = yf.download(Selected_stock,Dt1, Dt2)

#rest_index means- Does Not Mixed data When data is downloaded From yfinance and Save Data permanent.
data.reset_index(inplace=True)
data_load_state.text('Loading data... Done!')

#data.tail- Fetch data last days Latest Data..
st.subheader("Raw Data-")
st.write(data.tail())

#Plot Graph- go.Figure- figure class 
fig = go.Figure()
#Scatter Plot Draw. Open And close Stock Show Through Scatter Plot
fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
#Update Stocks Data Using Range Sliding Bar.
fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)
#Prediction--(Traning Data Freame)

df_train = data[['Date','Close']]

df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

#f is object of Prophet 
f = Prophet()
#fit- Data Train
f.fit(df_train)
#Data Frame Create for Future
future = f.make_future_dataframe(periods=period)
#Forecasting
forecast = f.predict(future) 

st.subheader('Forecast Data')
st.write(forecast.tail())
#Plot Graph For Prediction
st.write(f'Forecast Plot for {N_years} years')
fig1 = plot_plotly(f, forecast)
st.plotly_chart(fig1)
#fig1 and fig2 are Two figure
#Components Based Upon Forecasting Components
st.write("Forecost components")
fig2 = f.plot_components(forecast)
st.write(fig2)
