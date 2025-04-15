from flask import Flask, request, render_template_string
import pandas as pd
import pycaret.regression as pyc

app = Flask(__name__)

# Load trained model
model_pipeline = pyc.load_model('random_forest_regressor_model')

form_html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>CropYield AI</title>
  <link rel="icon" href="/static/images/favicon.png">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Smart agriculture tool for predicting crop yield using AI">
  <style>
    body {
      background-image: url('/static/images/farming_background.png');
      background-size: cover;
      background-position: center;
      font-family: Arial, sans-serif;
      color: #f2f2f2;
      margin: 0;
      padding: 0;
    }
    .form-container {
      background-color: rgba(20, 20, 20, 0.88);
      padding: 30px;
      max-width: 500px;
      margin: 80px 0 80px 80px;
      border-radius: 15px;
      box-shadow: 0 0 20px rgba(0, 0, 0, 0.4);
    }
    label {
      margin-top: 12px;
      display: block;
      font-size: 17px;
    }
    input[type="text"], input[type="number"], select {
      width: 100%;
      padding: 10px;
      margin: 6px 0 16px;
      border-radius: 6px;
      border: none;
      font-size: 17px;
      background-color: #f8f8f8;
      color: #333;
      box-sizing: border-box;
    }
    input[type="radio"] {
      margin-right: 5px;
    }
    .radio-group {
      display: flex;
      gap: 20px;
      margin-bottom: 16px;
    }
    button {
      padding: 12px;
      width: 100%;
      background-color: #2e7d32;
      color: #ffffff;
      border: none;
      font-size: 18px;
      font-weight: bold;
      border-radius: 8px;
      cursor: pointer;
      box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    button:hover {
      background-color: #1b5e20;
    }
  </style>
</head>
<body>
  <nav style="background-color: rgba(0, 0, 0, 0.85); padding: 10px 20px; color: #fff; font-size: 18px; display: flex; justify-content: space-between; align-items: center;">
    <div><strong>CropYield AI</strong></div>
    <div>
      <a href="/" style="color: #fff; margin-right: 15px; text-decoration: none;">Home</a>
      <a href="mailto:W0830731@myscc.ca" style="color: #fff; text-decoration: none;">Contact</a>
    </div>
  </nav>

  <div style="display: flex; justify-content: space-between; align-items: flex-start; padding: 40px; gap: 40px; flex-wrap: wrap;">
    <!-- LEFT: Form -->
    <div class="form-container">
      <h1 style='text-align: center;'>Crop Yield Prediction</h1>
      <p>Enter crop features manually or upload a CSV file with multiple entries.</p>
      <form method="POST" enctype="multipart/form-data">
        <label>Region:</label>
        <select name="Region">
          <option>North</option><option>South</option><option>East</option><option>West</option>
        </select>

        <label>Soil Type:</label>
        <select name="Soil_Type">
          <option>Clay</option><option>Loam</option><option>Sandy</option><option>Silt</option>
        </select>

        <label>Crop:</label>
        <select name="Crop">
          <option>Cotton</option><option>Rice</option><option>Soybean</option><option>Wheat</option>
        </select>

        <label>Rainfall (mm):</label>
        <input type="number" name="Rainfall_mm" step="0.1" required>

        <label>Temperature (°C):</label>
        <input type="number" name="Temperature_Celsius" step="0.1" required>

        <label>Fertilizer Used:</label>
        <div class="radio-group">
          <label><input type="radio" name="Fertilizer_Used" value="True" checked> Yes</label>
          <label><input type="radio" name="Fertilizer_Used" value="False"> No</label>
        </div>

        <label>Irrigation Used:</label>
        <div class="radio-group">
          <label><input type="radio" name="Irrigation_Used" value="True" checked> Yes</label>
          <label><input type="radio" name="Irrigation_Used" value="False"> No</label>
        </div>

        <label>Weather Condition:</label>
        <select name="Weather_Condition">
          <option>Rainy</option><option>Sunny</option>
        </select>

        <label>Days to Harvest:</label>
        <input type="number" name="Days_to_Harvest" required>

        <div style="margin: 20px 0; display: flex; align-items: center; gap: 10px;">
          <label style="margin: 0;">Or Upload CSV File:</label>
          <input type="file" name="file" style="padding: 6px 12px; border-radius: 6px; background-color: #ffffff; color: #000; border: none; font-size: 14px;">
        </div>

        <button type="submit">Predict</button>
        <br>
        <div id="result-message"></div>
      </form>
    </div>

    <!-- RIGHT: Home + Insights -->
    <div style="flex: 1; min-width: 360px; max-width: 520px; display: flex; flex-direction: column; gap: 20px;">
      <div style="background-color: rgba(0, 0, 0, 0.75); padding: 20px; border-radius: 12px; color: #f0f0f0;">
        <h2 style="font-size: 24px; color: #bbf7d0; margin-bottom: 10px;">Welcome to Yield Predict</h2>
        <p style="font-size: 17px; line-height: 1.6;">
          Yield Predict is a smart agricultural forecasting system that harnesses IoT and machine learning to predict crop yields with high accuracy. It empowers farms of all sizes to make sustainable, data-driven decisions.
        </p>
      </div>

      <div style="background-color: rgba(0, 0, 0, 0.75); padding: 24px; border-radius: 12px; color: #f0f0f0; font-size: 17px; line-height: 1.6;">
        <h2 style="color: #bbf7d0; font-size: 24px; margin-bottom: 10px;">🌾 Yield Predict – Smart Agriculture</h2>
        <p><strong>🔍 Goals:</strong> Maximize yield, reduce waste, and enable AI-powered decision-making.</p>
        <p><strong>🌱 Features:</strong><br>
          • Real-time IoT monitoring<br>
          • Predictive analytics with ML<br>
          • Smart fertilizer/irrigation suggestions<br>
          • Upload CSV for batch predictions<br>
          • Fully responsive on all devices
        </p>
        <p><strong>📈 Insights:</strong><br>
          • Key factors: Rainfall, Fertilizer, Temperature<br>
          • Accuracy: R² = 0.91 using Random Forest
        </p>
        <p><strong>📦 Training:</strong> PyCaret AutoML with Chi-Square, Forward Selection, and Random Forest</p>
        <p><strong>🛡️ Secure & Scalable</strong>: Compliant with standards and farm-ready</p>
      </div>
    </div>
  </div>

  <footer style="text-align: center; padding: 20px 0; font-size: 17.5px; color: #e6e6e6;">
    <p>
      <span style="background-color: #2e7d32; color: #ffffff; padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">© 2025 CropYield AI</span>
      <a href="mailto:W0830731@myscc.ca" style="margin: 0 8px; text-decoration: none; background-color: #2e7d32; color: white; padding: 8px 16px; border-radius: 6px;">Contact Us</a>
      <a href="https://github.com/Chhavinder058/CAPSTONE" target="_blank" style="text-decoration: none; background-color: #2e7d32; color: white; padding: 8px 16px; border-radius: 6px;">GitHub</a>
    </p>
  </footer>

  <script>
    const form = document.querySelector('form');
    const messageDiv = document.getElementById('result-message');
    form.addEventListener('submit', () => {
      messageDiv.innerHTML = '<p style="color:#ccc; text-align:center">⏳ Predicting... please wait.</p>';
    });
  </script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template_string(form_html)

    result_html = "<h2>Prediction Results</h2>"
    actual_y = None
    uploaded_file = request.files.get('file')

    if uploaded_file and uploaded_file.filename != '':
        df = pd.read_csv(uploaded_file)
        if 'Yield_tons_per_hectare' in df.columns:
            actual_y = df['Yield_tons_per_hectare']
            df.drop('Yield_tons_per_hectare', axis=1, inplace=True)
    else:
        data = {
            'Region': [request.form['Region']],
            'Soil_Type': [request.form['Soil_Type']],
            'Crop': [request.form['Crop']],
            'Rainfall_mm': [float(request.form['Rainfall_mm'])],
            'Temperature_Celsius': [float(request.form['Temperature_Celsius'])],
            'Fertilizer_Used': [request.form.get('Fertilizer_Used') == 'True'],
            'Irrigation_Used': [request.form.get('Irrigation_Used') == 'True'],
            'Weather_Condition': [request.form['Weather_Condition']],
            'Days_to_Harvest': [float(request.form['Days_to_Harvest'])]
        }
        df = pd.DataFrame(data)

    df = pd.get_dummies(df)

    model_features = [
        'Rainfall_mm', 'Temperature_Celsius', 'Fertilizer_Used', 'Irrigation_Used', 'Days_to_Harvest',
        'Region_North', 'Region_South', 'Region_West',
        'Soil_Type_Clay', 'Soil_Type_Loam', 'Soil_Type_Peaty', 'Soil_Type_Sandy', 'Soil_Type_Silt',
        'Crop_Cotton', 'Crop_Maize', 'Crop_Rice', 'Crop_Soybean', 'Crop_Wheat',
        'Weather_Condition_Rainy', 'Weather_Condition_Sunny'
    ]
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    df = df[model_features]

    predictions = model_pipeline.predict(df)

    if len(predictions) == 1:
        result_html += f"<p>Predicted Yield: <strong>{predictions[0]:.2f}</strong> tons per hectare</p>"
    else:
        result_html += f"<p>Predicted yields for <strong>{len(predictions)}</strong> entries:</p><table border='1'><tr>"
        if actual_y is not None:
            result_html += "<th>Index</th><th>Actual Yield</th><th>Predicted Yield</th></tr>"
        else:
            result_html += "<th>Index</th><th>Predicted Yield</th></tr>"
        for i in range(min(len(predictions), 10)):
            if actual_y is not None:
                result_html += f"<tr><td>{i}</td><td>{actual_y.iloc[i]:.2f}</td><td>{predictions[i]:.2f}</td></tr>"
            else:
                result_html += f"<tr><td>{i}</td><td>{predictions[i]:.2f}</td></tr>"
        result_html += "</table>"

    return result_html

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)