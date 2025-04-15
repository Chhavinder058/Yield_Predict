from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)
model = joblib.load('random_forest_regressor_model.pkl')

# Define expected feature columns after preprocessing (one-hot encoding with drop_first)
expected_cols = ['Rainfall_mm', 'Temperature_Celsius', 'Fertilizer_Used', 'Irrigation_Used', 
                 'Days_to_Harvest', 'Region_North', 'Region_South', 'Region_West', 
                 'Soil_Type_Loam', 'Soil_Type_Sandy', 'Soil_Type_Silt', 
                 'Crop_Cotton', 'Crop_Rice', 'Crop_Soybean', 'Crop_Wheat', 
                 'Weather_Condition_Rainy', 'Weather_Condition_Sunny']

# Approximated training means and stds for scaling (from Capstone data)
scaler_means = {
    'Rainfall_mm': 549.981901, 'Temperature_Celsius': 27.504965, 'Days_to_Harvest': 104.495025,
    'Fertilizer_Used': 0.5, 'Irrigation_Used': 0.5,
    'Region_North': 0.25, 'Region_South': 0.25, 'Region_West': 0.25,
    'Soil_Type_Loam': 0.25, 'Soil_Type_Sandy': 0.25, 'Soil_Type_Silt': 0.25,
    'Crop_Cotton': 0.2, 'Crop_Rice': 0.2, 'Crop_Soybean': 0.2, 'Crop_Wheat': 0.2,
    'Weather_Condition_Rainy': 0.3333, 'Weather_Condition_Sunny': 0.3333
}
scaler_stds = {
    'Rainfall_mm': 259.851320, 'Temperature_Celsius': 7.220608, 'Days_to_Harvest': 25.953412,
    'Fertilizer_Used': 0.5, 'Irrigation_Used': 0.5,
    'Region_North': np.sqrt(0.25*0.75), 'Region_South': np.sqrt(0.25*0.75), 'Region_West': np.sqrt(0.25*0.75),
    'Soil_Type_Loam': np.sqrt(0.25*0.75), 'Soil_Type_Sandy': np.sqrt(0.25*0.75), 'Soil_Type_Silt': np.sqrt(0.25*0.75),
    'Crop_Cotton': np.sqrt(0.2*0.8), 'Crop_Rice': np.sqrt(0.2*0.8), 'Crop_Soybean': np.sqrt(0.2*0.8), 'Crop_Wheat': np.sqrt(0.2*0.8),
    'Weather_Condition_Rainy': np.sqrt(0.3333*0.6667), 'Weather_Condition_Sunny': np.sqrt(0.3333*0.6667)
}

# HTML template for the input form
form_html = """
<h1>Crop Yield Prediction</h1>
<p>Enter crop features manually or upload a CSV file with multiple entries.</p>
<form method="POST" enctype="multipart/form-data">
  <label>Region:</label>
  <select name="Region">
    <option>East</option><option>North</option>
    <option>South</option><option>West</option>
  </select><br><br>
  <label>Soil Type:</label>
  <select name="Soil_Type">
    <option>Clay</option><option>Loam</option>
    <option>Sandy</option><option>Silt</option>
  </select><br><br>
  <label>Crop:</label>
  <select name="Crop">
    <option>Barley</option><option>Cotton</option>
    <option>Rice</option><option>Soybean</option><option>Wheat</option>
  </select><br><br>
  <label>Rainfall (mm):</label>
  <input type="number" name="Rainfall_mm" step="0.1" required><br><br>
  <label>Temperature (°C):</label>
  <input type="number" name="Temperature_Celsius" step="0.1" required><br><br>
  <label>Fertilizer Used:</label>
  <input type="radio" name="Fertilizer_Used" value="True" checked> Yes
  <input type="radio" name="Fertilizer_Used" value="False"> No<br><br>
  <label>Irrigation Used:</label>
  <input type="radio" name="Irrigation_Used" value="True" checked> Yes
  <input type="radio" name="Irrigation_Used" value="False"> No<br><br>
  <label>Weather Condition:</label>
  <select name="Weather_Condition">
    <option>Cloudy</option><option>Rainy</option><option>Sunny</option>
  </select><br><br>
  <label>Days to Harvest:</label>
  <input type="number" name="Days_to_Harvest" required><br><br>
  <label>Or Upload CSV File:</label>
  <input type="file" name="file"><br><br>
  <button type="submit">Predict</button>
</form>
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template_string(form_html)
    # POST: process form or file input
    result_html = "<h2>Prediction Results</h2>"
    actual_y = None

    # Check if a file was uploaded
    uploaded_file = request.files.get('file')
    if uploaded_file and uploaded_file.filename != '':
        # Read CSV into DataFrame
        df = pd.read_csv(uploaded_file)
        # If actual yield provided in file, separate it for residuals plot
        if 'Yield_tons_per_hectare' in df.columns:
            actual_y = df['Yield_tons_per_hectare']
            df.drop('Yield_tons_per_hectare', axis=1, inplace=True)
    else:
        # No file, use manual form inputs to create DataFrame
        data = {
            'Region': [request.form['Region']],
            'Soil_Type': [request.form['Soil_Type']],
            'Crop': [request.form['Crop']],
            'Rainfall_mm': [float(request.form['Rainfall_mm'])],
            'Temperature_Celsius': [float(request.form['Temperature_Celsius'])],
            'Fertilizer_Used': [True if request.form.get('Fertilizer_Used') == 'True' else False],
            'Irrigation_Used': [True if request.form.get('Irrigation_Used') == 'True' else False],
            'Weather_Condition': [request.form['Weather_Condition']],
            'Days_to_Harvest': [float(request.form['Days_to_Harvest'])]
        }
        df = pd.DataFrame(data)

    # Ensure boolean columns are of bool type (if they came in as strings from CSV)
    if 'Fertilizer_Used' in df.columns and df['Fertilizer_Used'].dtype == object:
        df['Fertilizer_Used'] = df['Fertilizer_Used'].map({'True': True, 'False': False})
        df['Irrigation_Used'] = df['Irrigation_Used'].map({'True': True, 'False': False})

    # One-hot encode categorical features
    df_proc = pd.get_dummies(df, drop_first=True)
    # Drop any unexpected dummy columns (categories not seen in training)
    for col in list(df_proc.columns):
        if col not in expected_cols:
            df_proc.drop(columns=[col], inplace=True)
    # Add missing dummy columns with 0
    for col in expected_cols:
        if col not in df_proc.columns:
            df_proc[col] = 0
    # Reorder columns to expected order
    df_proc = df_proc[expected_cols]
    # Scale features using training mean and std
    for col in expected_cols:
        df_proc[col] = (df_proc[col] - scaler_means[col]) / scaler_stds[col]

    # Predict yields
    predictions = model.predict(df_proc)
    # Format prediction results
    if len(predictions) == 1:
        result_html += f"<p>Predicted Yield: <strong>{predictions[0]:.2f}</strong> tons per hectare</p>"
    else:
        result_html += f"<p>Predicted yields for <strong>{len(predictions)}</strong> entries:</p>"
        result_html += "<table border='1'><tr>"
        if actual_y is not None:
            result_html += "<th>Index</th><th>Actual Yield</th><th>Predicted Yield</th></tr>"
        else:
            result_html += "<th>Index</th><th>Predicted Yield</th></tr>"
        # Show first 10 predictions in table (or all if fewer)
        n = len(predictions)
        limit = min(n, 10)
        for i in range(limit):
            if actual_y is not None:
                result_html += f"<tr><td>{i}</td><td>{actual_y.iloc[i]:.2f}</td><td>{predictions[i]:.2f}</td></tr>"
            else:
                result_html += f"<tr><td>{i}</td><td>{predictions[i]:.2f}</td></tr>"
        if n > limit:
            # indicate more entries
            extra_cols = 3 if actual_y is not None else 2
            result_html += "<tr>" + "<td>...</td>" * extra_cols + "</tr>"
        result_html += "</table>"

    # Generate Feature Importance plot
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # indices of features sorted by importance (desc)
    fig1 = plt.figure(figsize=(6,4))
    plt.barh(np.arange(len(importances)), importances[indices[::-1]], color='skyblue')
    plt.yticks(np.arange(len(importances)), np.array(expected_cols)[indices[::-1]])
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    # Encode plot to base64
    buf = BytesIO()
    fig1.savefig(buf, format='png')
    buf.seek(0)
    fi_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig1)
    result_html += f"<h3>Feature Importance</h3><img src='data:image/png;base64,{fi_image}' alt='Feature Importance Plot'/>"

    # Generate residual plot if actual values are available
    if actual_y is not None and len(actual_y) == len(predictions):
        fig2 = plt.figure(figsize=(5,5))
        plt.scatter(actual_y, predictions, alpha=0.5)
        min_val = min(actual_y.min(), predictions.min())
        max_val = max(actual_y.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel("Actual Yield")
        plt.ylabel("Predicted Yield")
        plt.title("Actual vs Predicted")
        plt.tight_layout()
        buf2 = BytesIO()
        fig2.savefig(buf2, format='png')
        buf2.seek(0)
        res_image = base64.b64encode(buf2.getvalue()).decode('utf-8')
        plt.close(fig2)
        result_html += f"<h3>Residuals (Actual vs Predicted)</h3><img src='data:image/png;base64,{res_image}' alt='Residual Plot'/>"

    return result_html

if __name__ == '__main__':
    app.run(debug=True)
    