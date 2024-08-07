---

# Weather Prediction Web Application

**Description:**
This web application predicts weather conditions based on user input using a machine learning model. The application is built with Streamlit and utilizes a trained XGBoost model to provide real-time weather predictions.

**Skills Taken From This Project:**
- Python scripting
- Data preprocessing
- Machine Learning
- Web application development with Streamlit
- Data visualization with Plotly

**Domain:**
Data Science and Machine Learning

**Problem Statement:**
The goal is to predict the weather conditions based on various meteorological and environmental parameters provided by the user. This helps users get real-time weather predictions and insights.

**Dataset Information:**
The dataset used in this project includes the following columns:
- Temperature
- Humidity
- Wind Speed
- Precipitation (%)
- Cloud Cover
- Atmospheric Pressure
- UV Index
- Visibility (km)
- Season
- Location
- Weather Type (target variable)

**Tasks Completed:**

**Task 1: Data Engineering**
1. Loaded and preprocessed the dataset.
2. Applied label encoding and scaling to prepare data for model training.
3. Saved the preprocessed data and model components for deployment.

**Task 2: Model Development**
1. Developed and trained an XGBoost model for predicting weather conditions.
2. Saved the trained model, scaler, and label encoders for use in the application.

**Task 3: Web Application Development**
1. Built a Streamlit application for user interaction and prediction.
2. Implemented functionality for:
   - User input for weather parameters.
   - Displaying predictions based on user input.
   - Visualizing dataset characteristics and distributions.
3. Deployed the application using Render.

**Features:**
- **Home Page:** Introduction to the application.
- **Prediction Tab:** Allows users to input weather parameters and view predictions.
- **Visualization Tab:** Provides various visualizations including:
  - Dataset Overview
  - Categorical Feature Distributions (Pie Charts)
  - Mean Values by Categorical Features (Bar Charts)
  - Feature Distributions (Histograms)

**Deployed Application:**
The application is deployed on Render. You can access it [here](https://weather-predictions.onrender.com).

**GitHub Repository:**
[Weather Prediction Web Application GitHub Repository](https://github.com/Cid-SK/Weather-Prediction-Web-Application.git)

**Setup Instructions:**

### Prerequisites
Ensure you have Python installed. Use the following commands to set up a virtual environment and install required packages:

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Project Structure
- `app.py`: The main Streamlit application script.
- `requirements.txt`: List of required Python packages.
- `xgb_model.pkl`: Trained XGBoost model.
- `scaler.pkl`: Scaler used for feature standardization.
- `reverse_mapping.pkl`: Mapping for reversing numerical predictions.
- `Cloud Cover_label_encoder.pkl`, `Season_label_encoder.pkl`, `Location_label_encoder.pkl`: Label encoders for categorical features.

### Running the Application
To run the application locally, use the following command:

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser to interact with the application.

**Configuration:**
The application uses the following configurations:
- **Model Path**: Path to the trained XGBoost model.
- **Scaler Path**: Path to the scaler for feature standardization.
- **Label Encoder Paths**: Paths to the label encoders for categorical features.
- **Reverse Mapping Path**: Path to the reverse mapping for predictions.

**Usage:**
1. **Home Page:** Provides an overview of the application.
2. **Prediction Tab:**
   - Input weather parameters using sliders and select boxes.
   - Click the button to get the prediction.
   - View the selected values and prediction on the same page.
3. **Visualization Tab:**
   - Explore the dataset, view distributions, and mean values.
   - Select features for visualizing distributions, categorical value counts, and mean values.

**Contributing:**
Feel free to submit issues and pull requests. Your contributions are welcome!

**Contact:**
For any questions or feedback, please contact [Sathish](mailto:2310sathishkumarsk@gmail.com).

---
