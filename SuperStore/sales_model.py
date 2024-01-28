import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
import os

sales = pd.read_excel('Sample - Superstore.xls')

sales['Order Date'] = pd.to_datetime(sales['Order Date'])

sales['Month'] = sales['Order Date'].dt.month_name()

new_sales = sales.drop(['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Customer ID', 'Customer Name', 'Postal Code',
                        'Product ID', 'Product Name', 'City'], axis = 1)

def check(df):
    for i in df.columns:
        if df[i].dtype == 'object':

            print(f'{i} : {df[i].unique()}')


check(new_sales)

def replace_values_with_dict(df, column_dict):
    for column, value_dict in column_dict.items():
        df[column] = df[column].replace(value_dict)
    return df

# Example usage for 'new_sales' DataFrame
column_dicts = {
    'Ship Mode': {'Second Class': 0, 'Standard Class': 1, 'First Class': 2, 'Same Day': 3},
    'Segment': {'Consumer': 0, 'Corporate': 1, 'Home Office': 2},
    'Country': {'United States': 1},
    'State': {'Kentucky': 0, 'California': 1, 'Florida': 2, 'North Carolina': 3, 'Washington': 4, 'Texas': 5,
              'Wisconsin': 6, 'Utah': 7, 'Nebraska': 8, 'Pennsylvania': 9, 'Illinois': 10, 'Minnesota': 11,
              'Michigan': 12, 'Delaware': 13, 'Indiana': 14, 'New York': 15, 'Arizona': 16, 'Virginia': 17,
              'Tennessee': 18, 'Alabama': 19, 'South Carolina': 20, 'Oregon': 21, 'Colorado': 22, 'Iowa': 23, 'Ohio': 24,
              'Missouri': 25, 'Oklahoma': 26, 'New Mexico': 27, 'Louisiana': 28, 'Connecticut': 29, 'New Jersey': 30,
              'Massachusetts': 31, 'Georgia': 32, 'Nevada': 33, 'Rhode Island': 34, 'Mississippi': 35,
              'Arkansas': 36, 'Montana': 37, 'New Hampshire': 38, 'Maryland': 39, 'District of Columbia': 40,
              'Kansas': 41, 'Vermont': 42, 'Maine': 43, 'South Dakota': 44, 'Idaho': 45, 'North Dakota': 46,
              'Wyoming': 47, 'West Virginia': 48},
    'Region': {'South': 0, 'West': 1, 'Central': 2, 'East': 3},
    'Category': {'Furniture': 0, 'Office Supplies': 1, 'Technology': 2},
    'Sub-Category': {'Bookcases': 0, 'Chairs': 1, 'Labels': 2, 'Tables': 3, 'Storage': 4, 'Furnishings': 5, 'Art': 6,
                     'Phones': 7, 'Binders': 8, 'Appliances': 9, 'Paper': 10, 'Accessories': 11, 'Envelopes': 12,
                     'Fasteners': 13, 'Supplies': 14, 'Machines': 15, 'Copiers': 16},
    'Month': {'November': 0, 'June': 1, 'October': 2, 'April': 3, 'December': 4, 'May': 5, 'August': 6, 'July': 7,
              'September': 8, 'January': 9, 'March': 10, 'February': 11}
}

new_sales = replace_values_with_dict(new_sales, column_dicts)

x = new_sales.drop('Profit', axis = 1)
y = new_sales['Profit']

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
#from catboost import CatBoostRegressor
#from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
import os
import joblib
import pickle
import xgboost as xgb

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

model = XGBRegressor(n_estimaators = 100, learning_rate = 0.2)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

predictions = pd.DataFrame({'Actual Values': y_test, 'Predicted Values': y_pred})

print(predictions)

# Save the model to a file
with open('newest_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully.")



    