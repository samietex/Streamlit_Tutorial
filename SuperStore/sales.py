import streamlit as st
import pickle
import joblib
import numpy as np
import plotly.express as px
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Superstore!!!", page_icon=":bar_chart:",layout="wide")

st.title(" :bar_chart: Sample SuperStore EDA")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

# Get the directory of the script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the file path relative to the script directory
file_path = os.path.join(script_directory, 'Sample - Superstore.xls')

# Check if the file exists before reading it
if os.path.exists(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)
    st.write('File loaded successfully!')
    st.write(df.head())  # Print a sample of the data
else:
    st.error('File not found. Please check the file path.')

df["Order Date"] = pd.to_datetime(df["Order Date"])

df["month"] = df["Order Date"].dt.month_name()  # Extract month name

# Now convert 'Order Date' to string format for filtering
df['Order Date'] = df['Order Date'].dt.strftime('%Y-%m-%d')

# Getting the min and max date
startDate = df["Order Date"].min()
endDate = df["Order Date"].max()


col1, col2 = st.columns((2,2))

with col1:
    date1 = st.date_input('Start Date', pd.to_datetime(startDate)).strftime('%Y-%m-%d')

with col2:
    date2 = st.date_input('End Date', pd.to_datetime(endDate)).strftime('%Y-%m-%d')

df = df[(df['Order Date'] >= date1) & (df['Order Date'] <= date2)].copy()

with st.sidebar:
    st.header("Choose your filter: ")
    region = st.multiselect('Region', df['Region'].unique())
    if not region:
        df2 = df.copy()
    else:
        df2 = df[df["Region"].isin(region)]
    
    state = st.multiselect('State', df2['State'].unique())
    if not state:
        df3 = df2.copy()
    else:
        df3 = df2[df2['State'].isin(state)]
    
    city = st.multiselect('City', df3['City'].unique())
    
    if not region and not state and not city:
        filtered_df = df
    elif not state and not city:
        filtered_df = df[df["Region"].isin(region)]
    elif not region and not city:
        filtered_df = df[df["State"].isin(state)]
    elif state and city:
        filtered_df = df3[df["State"].isin(state) & df3["City"].isin(city)]
    elif region and city:
        filtered_df = df3[df["Region"].isin(region) & df3["City"].isin(city)]
    elif region and state:
        filtered_df = df3[df["Region"].isin(region) & df3["State"].isin(state)]
    elif city:
        filtered_df = df3[df3["City"].isin(city)]
    else:
        filtered_df = df3[df3["Region"].isin(region) & df3["State"].isin(state) & df3["City"].isin(city)]

    category_df = filtered_df.groupby(by = ["Category"], as_index = False)["Sales"].sum()


with col1:
    st.subheader("Category wise Sales")
    fig = px.bar(category_df, x = "Category", y = "Sales", text = ['${:,.2f}'.format(x) for x in category_df["Sales"]],
                 template = "seaborn")
    st.plotly_chart(fig,use_container_width=True, height = 200)

with col2:
    st.subheader('Region wise Sales')
    fig = px.pie(filtered_df, values = 'Sales', names = 'Region', hole = 0.5)
    fig.update_traces(text = filtered_df["Region"], textposition = "outside")
    st.plotly_chart(fig,use_container_width=True)

new_col1, new_col2 = st.columns((2))

with new_col1:
    with st.expander('Category_viewData'):
        st.write(category_df.style.background_gradient(cmap="Blues"))
        st.download_button('Download Data', data = 'csv', file_name = 'Category.csv', 
                           help = 'Click here to download the data as a CSV file')
        
with new_col2:
    with st.expander('Region_ViewData'):
        region = filtered_df.groupby(by = "Region", as_index = False)["Sales"].sum()
        st.write(region.style.background_gradient(cmap="Oranges"))
        csv = region.to_csv(index = False).encode('utf-8')
        st.download_button("Download Data", data = csv, file_name = "Region.csv", mime = "text/csv",
                        help = 'Click here to download the data as a CSV file')

filtered_df["month_year"] = pd.to_datetime(filtered_df["Order Date"]).dt.to_period("M")
st.subheader('Time Series Analysis')


linechart = pd.DataFrame(filtered_df.groupby("month_year")["Sales"].sum()).reset_index()
linechart['month_year'] = linechart['month_year'].dt.strftime('%Y-%m')


series1, series2 = st.columns((2))

with series1:

    button1 =  st.button('Time Series Line Chart')

with series2:
    button2 =  st.button('Time Series Bar Chart')

barchart = pd.DataFrame(filtered_df.groupby("month_year")["Sales"].sum()).reset_index()
barchart['month_year'] = barchart['month_year'].dt.strftime('%Y-%m')

if button1:
    fig2 = px.line(linechart, x = "month_year", y="Sales", labels = {"Sales": "Amount"},height=500, width = 1000,template="gridon")
    st.plotly_chart(fig2,use_container_width=True)

elif button2:
    fig_bar = px.bar(barchart, x="month_year", y="Sales", labels={"Sales": "Amount"}, height=500, width=1000, template="gridon")
    st.plotly_chart(fig_bar, use_container_width=True)

else:
    fig2 = px.line(linechart, x = "month_year", y="Sales", labels = {"Sales": "Amount"},height=500, width = 1000,template="gridon")
    st.plotly_chart(fig2,use_container_width=True)



with st.expander('View Data of TimeSeries:'):
    st.write(linechart.T.style.background_gradient(cmap = 'Blues'))
    csv = linechart.to_csv(index = False).encode('utf-8')
    st.download_button('Download Data', data = csv, file_name = 'TimeSeries.csv', mime = 'text/csv')

st.subheader('Hierarchical view of Sales using TreeMap')
fig3 = px.treemap(filtered_df, path = ['Region', 'Category', 'Sub-Category'], values = 'Sales', hover_data = ['Sales'],
                  color = 'Sub-Category')
fig3.update_layout(width = 800, height = 650)
st.plotly_chart(fig3, use_container_width=True)

chart1, chart2 = st.columns((2))
with chart1:
    st.subheader('Segment wise Sales')
    fig = px.pie(filtered_df, values = "Sales", names = "Segment", template = "plotly_dark")
    fig.update_traces(text = filtered_df["Segment"], textposition = "inside")
    st.plotly_chart(fig,use_container_width=True)

with chart2:
    st.subheader('Category wise Sales')
    fig = px.pie(filtered_df, values = "Sales", names = "Category", template = "gridon")
    fig.update_traces(text = filtered_df["Category"], textposition = "inside")
    st.plotly_chart(fig,use_container_width=True)

import plotly.figure_factory as ff
st.subheader(":point_right: Month wise Sub-Category Sales Summary")
with st.expander("Summary_Table"):
    df_sample = df[0:5][["Region","State","City","Category","Sales","Profit","Quantity"]]
    fig = ff.create_table(df_sample, colorscale = "Cividis")
    st.plotly_chart(fig, use_container_width=True)

# When you need to extract the month name, convert 'Order Date' back to datetime temporarily
filtered_df["month"] = pd.to_datetime(filtered_df["Order Date"]).dt.month_name()

# Now you can create the pivot table
sub_category_Year = pd.pivot_table(data=filtered_df, values="Sales", index=["Sub-Category"], columns="month")

# And then display it
st.markdown("Month wise sub-Category Table")
st.write(sub_category_Year.style.background_gradient(cmap="Blues"))


# Create a scatter plot
data1 = px.scatter(filtered_df, x = "Sales", y = "Profit", size = "Quantity")
data1['layout'].update(title = "Relationship between Sales and Profits using Scatter Plot.",
                       titlefont = dict(size = 20),xaxis = dict(title = "Sales", titlefont = dict(size = 19)),
                       yaxis = dict(title = "Profit", titlefont = dict(size = 19)))
st.plotly_chart(data1, use_container_width = True)

with st.expander("View Data"):
    st.write(filtered_df.iloc[:500,1:20:2].style.background_gradient(cmap="Oranges"))

# Download orginal DataSet
csv = df.to_csv(index = False).encode('utf-8')
st.download_button('Download Data', data = csv, file_name = "Data.csv",mime = "text/csv")



#-------THE PREDICTION APP------------
import streamlit as st
import pandas as pd
from joblib import load
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
#from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
#from catboost import CatBoostRegressor
#from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import os
import joblib
import pickle
#import xgboost as xgb

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

model = RandomForestRegressor(n_estimators = 100, max_depth = 10)

model.fit(x_train, y_train)


# Function to replace values in DataFrame columns
def replace_values_with_dict(df, column_dict):
    for column, value_dict in column_dict.items():
        df[column] = df[column].replace(value_dict)
    return df

st.title("Profit Prediction")

# Input features based on the dataset
col1, col2 = st.columns((2))

with col1:
    shipmode = st.selectbox('Ship Mode', ['Second Class', 'Standard Class', 'First Class', 'Same Day'])
    segment = st.selectbox('Segment', ['Consumer', 'Corporate', 'Home Office'])
    country = st.selectbox('Country', ['United States'])
    state = st.selectbox('State', ['Kentucky', 'California', 'Florida', 'North Carolina', 'Washington', 'Texas',
                                    'Wisconsin', 'Utah', 'Nebraska', 'Pennsylvania', 'Illinois', 'Minnesota',
                                    'Michigan', 'Delaware', 'Indiana', 'New York', 'Arizona', 'Virginia',
                                    'Tennessee', 'Alabama', 'South Carolina', 'Oregon', 'Colorado', 'Iowa', 'Ohio',
                                    'Missouri', 'Oklahoma', 'New Mexico', 'Louisiana', 'Connecticut', 'New Jersey',
                                    'Massachusetts', 'Georgia', 'Nevada', 'Rhode Island', 'Mississippi',
                                    'Arkansas', 'Montana', 'New Hampshire', 'Maryland', 'District of Columbia',
                                    'Kansas', 'Vermont', 'Maine', 'South Dakota', 'Idaho', 'North Dakota',
                                    'Wyoming', 'West Virginia'])
    region = st.selectbox('Region', ['South', 'West', 'Central', 'East'])

with col2:
    category = st.selectbox('Category', ['Furniture', 'Office Supplies', 'Technology'])
    sub_category = st.selectbox('Sub-Category', ['Bookcases', 'Chairs', 'Labels', 'Tables', 'Storage', 'Furnishings', 'Art',
                                                  'Phones', 'Binders', 'Appliances', 'Paper', 'Accessories', 'Envelopes',
                                                  'Fasteners', 'Supplies', 'Machines', 'Copiers'])
    sales = st.slider('Sales', min_value=0, max_value=30000, step=1)
    quantity = st.slider('Quantity', min_value=1, max_value=20, step=1)
    discount = st.slider('Discount', min_value=0, max_value=1)
    month = st.selectbox('Month', ['November', 'June', 'October', 'April', 'December', 'May', 'August', 'July',
                                    'September', 'January', 'March', 'February'])

if st.button('Predict Profit'):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame([[shipmode, segment, country, state, region, category, sub_category, sales, quantity, discount, month]],
                              columns=['Ship Mode', 'Segment', 'Country', 'State', 'Region', 'Category', 'Sub-Category', 'Sales', 'Quantity', 'Discount', 'Month'])

    st.write('Raw input data:', input_data)

    # Define column dictionaries for replacement
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

    # Replace values in the input data
    input_data_processed = replace_values_with_dict(input_data, column_dicts)

    st.write('Processed input data:', input_data_processed)



    predictions = model.predict(input_data_processed)[0]

    st.write(f'The predicted profit is {predictions}')



