# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 22:50:43 2022

@author: ashwi
"""
import pandas as pd
df = pd.read_excel("CocaCola_Sales_Rawdata.xlsx")
df
df.shape
df.dtypes

# finding the null values
df.isnull().sum()
# there is no null values


df["Sales"].plot()
# there is upword linear trend with multiplicative seasonality

# Trend,seasonal,residual,observed graphs 
import statsmodels.api as smf
from statsmodels.tsa.seasonal import seasonal_decompose
seasonal_ts_add = smf.tsa.seasonal_decompose(df["Sales"],extrapolate_trend='freq',period=10)
seasonal_ts_add.plot()

# boxplot
import seaborn as sns
sns.boxplot(data=df["Sales"])


# Extracting quarter from Quarter

df['Quarter']
quarter = ['Q1','Q2','Q3','Q4']


p = df["Quarter"][0]
p[0:2]
df['quarter']= 0

for i in range(42):
    p = df["Quarter"][i]
    df['quarter'][i]= p[0:2]

df.head()


# extracting the dummies from the column Month
quarter_dummies = pd.DataFrame(pd.get_dummies(df["quarter"]))

df1 = pd.concat([df,quarter_dummies],axis=1)
df1

# creating new columns
import numpy as np
df1["t"] = np.arange(1,43)
df1["t_squared"] = df1["t"]*df1["t"]
df1["log_Sales"] = np.log(df1["Sales"])
df1

# Spliting the data into train and test
Train = df1.head(32)
Test = df1.tail(10)

# Linear Model
import statsmodels.formula.api as smf
linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
print("RMSE Linear: ",rmse_linear)

# Exponential Model
exp_model = smf.ols("log_Sales~t",data=Train).fit()
pred_exp = pd.Series(exp_model.predict(pd.DataFrame(Test['t'])))
rmse_exp =np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_exp)))**2))
print("RMSE Exponential: ",rmse_exp)

# Quadratic Model
quad_model = smf.ols("Sales~t+t_squared",data=Train).fit()
pred_quad = pd.Series(quad_model.predict(Test[['t','t_squared']]))
rmse_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_quad))**2))
print("RMSE Quadratic: ",rmse_quad)

# Additive Seasonality
add_sea = smf.ols("Sales~Q1+Q2+Q3+Q4",data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))  
print("RMSE Additive Seasonality: ",rmse_add_sea)

# Additive seasonality with Quadratic Trend
add_sea_quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4',data=Train).fit()
pred_ass_sea_quad = pd.Series(add_sea_quad.predict(Test))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_ass_sea_quad))**2))
print('RMSE Additive Seasonality Quadratic: ',rmse_add_sea_quad)

# Multiplicative Seasonality 
mul_sea = smf.ols('log_Sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_mul_sea = pd.Series(mul_sea.predict(Test))
rmse_mul_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_mul_sea)))**2))
print("RMSE Multiplicative Seasonality: ",rmse_mul_sea)


# Multiplicative Aditive Seasonality
mul_add_sea = smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data=Train).fit()
pred_mul_add_sea = pd.Series(mul_add_sea.predict(Test))
rmse_mul_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_mul_add_sea)))**2))
print("RMSE MUltiplicative Additive Seasonality: ",rmse_mul_add_sea)

# Creating table of RMSE values
data1 = {"MODEL":pd.Series(["RMSE Linear","RMSE Exponential","RMSE Quadratic","RMSE Additive Seasonality","RMSE Additive Seasonality Quadratic","RMSE Multiplicative Seasonality","RMSE MUltiplicative Additive Seasonality"]),
         "RMSE Value":pd.Series([rmse_linear,rmse_exp,rmse_quad,rmse_add_sea,rmse_add_sea_quad,rmse_mul_sea,rmse_mul_add_sea])}

table_rmse =pd.DataFrame(data1)
table_rmse
# therefore above different Forecasting Methods we are getting best result from the "Additive Seasonality Quadratic"

# Creating the new data to predict the future prediction
data = [['Q3_96', 'Q3'], ['Q4_96', 'Q4'], ['Q1_97', 'Q1'],['Q2_97', 'Q2']]
print(data)
forecast = pd.DataFrame(data, columns = ['Quarter', 'quarter'])
forecast

# creating dummies, t and t-squared columns

dummies = pd.DataFrame(pd.get_dummies(forecast['quarter']))         
forecast1 = pd.concat([forecast,dummies],axis = 1)
forecast1["t"] = np.arange(1,5)   
forecast1["t_squared"] = forecast1["t"]*forecast1["t"] 
print("\nAfter Dummy, T and T-Square\n\n",forecast1.head())


# forecasting Additive Seasonality Quadratic Model
model_full = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4',data=df1).fit()
pred_new  = pd.Series(model_full.predict(forecast1))
pred_new
forecast1["forecasted_sales"] = pd.Series(pred_new)

# Final Prediction for next 4 Quarters
Final_predict = forecast1.loc[:, ['Quarter', 'forecasted_sales']]
Final_predict
























