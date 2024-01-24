import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Superstore!!!", page_icon=":bar_chart:",layout="wide")

st.title(" :bar_chart: Sample SuperStore EDA")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

df = pd.read_excel('Sample - Superstore.xls')

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