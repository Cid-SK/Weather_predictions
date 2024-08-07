---

#**Weather Prediction Web Application**

**Overview**

This web application predicts weather conditions based on user input using a machine learning model. Built with Streamlit, it utilizes a trained XGBoost model to provide real-time weather predictions.

**Features**

- **Home Page:** Introduction to the application and its purpose.
- **Prediction Tab:** Allows users to input weather-related parameters and receive a prediction.
- **Visualization Tab:** Provides various visualizations, including:
- **Dataset Overview:** Displays the dataset and its summary.
- **Value Counts Pie Chart:** Visualizes distributions of categorical features.
- **Mean Values Bar Chart:** Shows mean values for selected features grouped by categorical variables.
- **Distribution Plot:** Displays the distribution of selected features with customizable colors.

**Setup**

**Prerequisites**

Ensure you have Python and the necessary packages installed. Create a virtual environment and install the dependencies using:

```bash
python -m venv venv
source venv/bin/activate  # For Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

**Project Structure**

- `app.py`: The main Streamlit application script.
- `requirements.txt`: List of required Python packages.
- `xgb_model.pkl`: Trained XGBoost model.
- `scaler.pkl`: Scaler used for feature standardization.
- `reverse_mapping.pkl`: Mapping for reversing numerical predictions.
- `Cloud Cover_label_encoder.pkl`, `Season_label_encoder.pkl`, `Location_label_encoder.pkl`: Label encoders for categorical features.

**Running the Application**

To run the application locally, use the following command:

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser to interact with the application.

**Deployment**

The application is deployed on [Render](https://render.com). You can access it [here](https://weather-predictions.onrender.com/).

**Configuration**

The application uses the following configurations:

- **Model Path:** Path to the trained XGBoost model.
- **Scaler Path:** Path to the scaler for feature standardization.
- **Label Encoder Paths:** Paths to the label encoders for categorical features.
- **Reverse Mapping Path:** Path to the reverse mapping for predictions.

**Usage**

1. **Home Page:** Provides an overview of the application.
2. **Prediction Tab:**
   - Input weather parameters using sliders and select boxes.
   - Click the button to get the prediction.
   - View the selected values and prediction on the same page.
3. **Visualization Tab:**
   - Explore the dataset, view distributions, and mean values.
   - Select features to visualize distributions, categorical value counts, and mean values.

**Contributing**

Feel free to submit issues and pull requests. Your contributions are welcome!

**Contact**

For any questions or feedback, please contact Sathish at 2310sathishkumarsk@gmail.com.

---
