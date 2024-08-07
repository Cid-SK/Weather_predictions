import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv("G:/SK/Ds/weather/weather_classification_data.csv")

# Load LabelEncoders
le_dict = {}
for col in ['Cloud Cover', 'Season', 'Location']:
    with open(f'G:/SK/Ds/weather/{col}_label_encoder.pkl', 'rb') as file:
        le_dict[col] = pickle.load(file)

# Load Scaler
with open('G:/SK/Ds/weather/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load Model
with open('G:/SK/Ds/weather/xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load reverse mappings
with open('G:/SK/Ds/weather/reverse_mapping.pkl', 'rb') as file:
    reverse_mapping = pickle.load(file)

# Streamlit page configuration
st.set_page_config(page_title="Weather Prediction", layout="wide")
st.title("Weather Prediction")
st.markdown("<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["Home", "Prediction", "Visualization"])

# Home Tab
with tab1:
    st.subheader(":blue[Welcome to the Weather Prediction Application]")
    st.write(
        "This application is designed to predict weather conditions based on user-inputted data. Utilizing advanced machine learning models, it provides insights into possible weather scenarios based on various meteorological features."
    )
    
    st.write(
        "### :blue[Features:]"
    )
    st.write(
        "- **User-Friendly Interface:** Input your weather data through sliders and dropdowns to get instant predictions."
    )
    st.write(
        "- **Prediction Model:** Leverages a trained XGBoost model to predict weather conditions based on historical data."
    )
    st.write(
        "- **Data Visualization:** Explore detailed visualizations of the dataset, including distribution and mean values of different features."
    )
    
    st.write(
        "### :blue[Technologies Used:]"
    )
    st.write(
        "- **Streamlit:** For building the interactive web application."
    )
    st.write(
        "- **XGBoost:** For the predictive modeling."
    )
    st.write(
        "- **Plotly:** For interactive visualizations."
    )
    
    st.write(
        "### :blue[How to Use:]"
    )
    st.write(
        "1. Navigate to the **Prediction** tab to input weather data and receive predictions."
    )
    st.write(
        "2. Use the **Visualization** tab to explore and visualize different aspects of the dataset."
    )
    

# Prediction Tab
with tab2:
    
    def user_input_features():
        c1, c2, c3 = st.columns(3)
        with c1:
            Temperature = st.slider('Temperature', min_value=-25, max_value=70, value=20)
            Humidity = st.slider('Humidity', min_value=20, max_value=110, value=50)
            Cloud_Cover = st.selectbox('Cloud Cover', ['partly cloudy', 'clear', 'overcast', 'cloudy'])
        with c2:
            Wind_Speed = st.slider('Wind Speed', min_value=0, max_value=27, value=10)
            Precipitation = st.slider('Precipitation (%)', min_value=0, max_value=100, value=55)
            Season = st.selectbox('Season', ['Winter', 'Spring', 'Summer', 'Autumn'])
        with c3:
            UV_Index = st.slider('UV Index', min_value=0, max_value=14, value=7)
            Visibility = st.slider('Visibility', min_value=0, max_value=15, value=10)
            Location = st.selectbox('Location', ['inland', 'mountain', 'coastal'])
 
        data = {
            'Temperature': Temperature,
            'Humidity': Humidity,
            'Wind Speed': Wind_Speed,
            'Precipitation (%)': Precipitation,
            'Cloud Cover': Cloud_Cover,
            'UV Index': UV_Index,
            'Visibility (km)': Visibility,
            'Season': Season,
            'Location': Location
        }
        
        return pd.DataFrame(data, index=[0])

    # Get user input
    sel_df = user_input_features()

    # Encode categorical features
    df_encoded = sel_df.copy()
    df_encoded['Cloud Cover'] = le_dict['Cloud Cover'].transform(df_encoded['Cloud Cover'])
    df_encoded['Season'] = le_dict['Season'].transform(df_encoded['Season'])
    df_encoded['Location'] = le_dict['Location'].transform(df_encoded['Location'])

    # Ensure columns match the order used during scaler fitting
    feature_cols = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Cloud Cover',
                    'UV Index', 'Season', 'Visibility (km)', 'Location']
    df_encoded = df_encoded[feature_cols]

    # Standardize the features
    df_scaled = scaler.transform(df_encoded)

    # Make prediction
    prediction = model.predict(df_scaled)

    # Map numerical prediction to categorical value
    prediction_label = reverse_mapping['Weather Type'][prediction[0]]

    st.write("Selected Values:")
    st.dataframe(sel_df)

    st.markdown(
        f"<div style='text-align:center; font-size:20px;'>"
        f"<span>Prediction:</span> "
        f"<span style='color:red;'>{prediction_label}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

# Visualization Tab
with tab3:

    with st.container(border=True):
        st.subheader(":blue[Dataset Overview:]")
        st.dataframe(df)

    with st.container(border=True):
        st.subheader(":blue[Dataset Summary:]")
        st.write(df.describe())

    with st.container(border=True):
        st.subheader(":blue[Select a feature for value counts pie chart:]")
        pie_feature = st.selectbox(
            'Select Categorical Feature for Pie Chart',
            ['Season', 'Weather Type', 'Location', 'Cloud Cover']
        )
        
        if pie_feature:
            value_counts = df[pie_feature].value_counts()
            fig = px.pie(value_counts, values=value_counts.values, names=value_counts.index,
                         title=f'Distribution of {pie_feature}',
                         labels={pie_feature: pie_feature})
            st.plotly_chart(fig)

    with st.container(border=True):
        st.subheader(":blue[Mean Values by Feature]")
        mean_feature = st.selectbox(
            'Select Feature for Mean Values',
            ['Season', 'Weather Type', 'Location', 'Cloud Cover']
        )

        if mean_feature:
            mean_values = df.groupby(mean_feature)[['Temperature', 'Precipitation (%)', 'Humidity', 'Wind Speed', 'Visibility (km)']].mean()
            mean_values_reset = mean_values.reset_index()
            fig = px.bar(mean_values_reset, x=mean_feature, y=['Temperature', 'Precipitation (%)', 'Humidity', 'Wind Speed', 'Visibility (km)'],
                        title=f'Mean Values Grouped by {mean_feature}',
                        labels={'value': 'Mean Value', mean_feature: mean_feature},
                        barmode='group')
            st.plotly_chart(fig)

    with st.container(border=True):
        st.subheader(":blue[Distribution of Selected Feature:]")
        selected_feature = st.selectbox(
            'Select Feature',
            ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Visibility (km)']
        )
        col1, col2 = st.columns([8, 2])
        with col2:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            # Color picker for histogram
            histogram_color = st.color_picker('Select Histogram Color', '#1f77b4')
        with col1:
            if selected_feature:
                fig = px.histogram(df, x=selected_feature, nbins=30,
                                   title=f'Distribution of {selected_feature}',
                                   labels={selected_feature: selected_feature})
                fig.update_traces(marker_color=histogram_color)
                st.plotly_chart(fig)
