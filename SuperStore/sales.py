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





def load_model(model_name, directory = 'saved_models'):
    filename = os.path.join(directory, f'{model_name}.pkl')
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def load_label_encoders(directory='saved_models'):
    encoders = {}
    for item in os.listdir(directory):
        if item.endswith('_encoder.pkl'):
            col = item.replace('_encoder.pkl', '')
            encoders[col] = joblib.load(os.path.join(directory, item))
    return encoders

from sklearn.preprocessing import LabelEncoder

def preprocess_input(df):
    for i in df.columns:
        if df[i].dtype == 'O':
            lb = LabelEncoder()
            df[i] = lb.fit_transform(df[i])
            
    return df

from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()

def scaler(df):
    # If exclude_cols is not provided, initialize it as an empty list
    #if exclude_cols is None:
        #exclude_cols = []

    # Get a list of numeric columns excluding the specified ones
    #numeric_cols = [col for col in df.columns if col not in exclude_cols]

    # Scale only the numeric columns
    for i in df.columns:
        df[[i]] = mm.fit_transform(df[[i]])
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
    sales = st.slider('Sales', min_value = 0, max_value = 30000, step = 1)
    quantity = st.slider('Quantity', min_value = 1, max_value = 20, step = 1)
    discount = st.slider('Discount', min_value = 0, max_value = 1)
    month = st.selectbox('Month', ['November', 'June', 'October', 'April', 'December', 'May', 'August', 'July',
                        'September', 'January', 'March', 'February'])

if st.button('Predict Profit'):
        # Create a DataFrame with the input features
        input_data = pd.DataFrame([[shipmode, segment, country, state, region, category, sub_category, sales, quantity, discount, month]],
                                  columns=['Ship Mode', 'Segment', 'Country', 'State', 'Region', 'Category', 'Sub-Category', 'Sales', 'Quantity', 'Discount', 'Month'])

        st.write('Raw input data:', input_data)

        # Load encoders
        #encoders = load_label_encoders()

        # Preprocess the input data
        input_data = preprocess_input(input_data)

        # Exclude the columns that are already encoded
        #exclude_cols = ['shipmode', 'segment', 'country', 'state', 'region', 'category', 'sub_category', 'month']

        # Scale the input data
        final_data = scaler(input_data)

        st.write(final_data)

        # Load the model (adjust the model name if needed)
        model = load_model('XGBoost')

        # Make prediction
        prediction = model.predict(final_data)
        prediction = prediction[0]

        # Display the prediction
        st.write(f'The predicted profit is {prediction}')  