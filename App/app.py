import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np


### CONFIG
st.set_page_config(
    page_title="E-commerce",
    page_icon="💸",
    layout="wide"
  )

### TITLE AND TEXT
st.title("Build dashboards with Streamlit 🎨")

st.markdown("""
    Welcome to this awesome `streamlit` dashboard. This library is great to build very fast and
    intuitive charts and application running on the web. Here is a showcase of what you can do with
    it. Our data comes from an e-commerce website that simply displays samples of customer sales. Let's check it out.
    Also, if you want to have a real quick overview of what streamlit is all about, feel free to watch the below video 👇
""")

### LOAD AND CACHE DATA
DATA_URL = ('https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/e-commerce_data.csv')

@st.cache # this lets the 
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    data["Date"] = data["Date"].apply(lambda x: pd.to_datetime(",".join(x.split(",")[-2:])))
    data["currency"] = data["currency"].apply(lambda x: pd.to_numeric(x[1:]))
    return data

data_load_state = st.text('Loading data...')
data = load_data(1000)
data_load_state.text("") # change text from "Loading data..." to "" once the the load_data function has run

## Run the below code if the check is checked ✅
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data) 


### SHOW GRAPH STREAMLIT

currency_per_country = data.set_index("country")["currency"]
st.bar_chart(currency_per_country)

### SHOW GRAPH PLOTLY + STREAMLIT

st.subheader("Simple bar chart built with Plotly")
st.markdown("""
    Now, the best thing about `streamlit` is its compatibility with other libraries. For example, you
    don't need to actually use built-in charts to create your dashboard, you can use :
    
    * [`plotly`](https://docs.streamlit.io/library/api-reference/charts/st.plotly_chart) 
    * [`matplotlib`](https://docs.streamlit.io/library/api-reference/charts/st.pyplot)
    * [`bokeh`](https://docs.streamlit.io/library/api-reference/charts/st.bokeh_chart)
    * ...
    This way, you have all the flexibility you need to build awesome dashboards. 🥰
""")
fig = px.histogram(data.sort_values("country"), x="country", y="currency", barmode="group")
st.plotly_chart(fig, use_container_width=True)


### SIDEBAR
st.sidebar.header("Build dashboards with Streamlit")
st.sidebar.markdown("""
    * [Load and showcase data](#load-and-showcase-data)
    * [Charts directly built with Streamlit](#simple-bar-chart-built-directly-with-streamlit)
    * [Charts built with Plotly](#simple-bar-chart-built-with-plotly)
    * [Input Data](#input-data)
""")
e = st.sidebar.empty()
e.write("")
st.sidebar.write("Made with 💖 by [Jedha](https://jedha.co)")

### EXPANDER

with st.expander("⏯️ Watch this 15min tutorial"):
    st.video("https://youtu.be/B2iAodr0fOo")

st.markdown("---")

#### CREATE TWO COLUMNS
col1, col2 = st.columns(2)

with col1:
        st.markdown("**1️⃣ Example of input widget**")
        country = st.selectbox("Select a country you want to see all time sales", data["country"].sort_values().unique())
        
        country_sales = data[data["country"]==country]
        fig = px.histogram(country_sales, x="Date", y="currency")
        fig.update_layout(bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**2️⃣ Example of input form**")

    with st.form("average_sales_per_country"):
        country = st.selectbox("Select a country you want to see sales", data["country"].sort_values().unique())
        start_period = st.date_input("Select a start date you want to see your metric")
        end_period = st.date_input("Select an end date you want to see your metric")
        submit = st.form_submit_button("submit")

        if submit:
            avg_period_country_sales = data[(data["country"]==country)]
            start_period, end_period = pd.to_datetime(start_period), pd.to_datetime(end_period)
            mask = (avg_period_country_sales["Date"] > start_period) & (avg_period_country_sales["Date"] < end_period)
            avg_period_country_sales = avg_period_country_sales[mask].mean()
            st.metric("Average sales during selected period (in $)", np.round(avg_period_country_sales, 2))