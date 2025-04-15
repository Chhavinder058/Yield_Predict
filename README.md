# CropYield AI

CropYield AI is a smart agricultural forecasting system that leverages IoT and machine learning to predict crop yields with high accuracy. This project is designed to empower farms of all sizes to make sustainable, data-driven decisions.

## Features

- **Real-time IoT Monitoring**: Integrates IoT data for real-time insights.
- **Predictive Analytics**: Uses machine learning models to predict crop yields.
- **Smart Suggestions**: Provides recommendations for fertilizer and irrigation usage.
- **Batch Predictions**: Supports CSV uploads for batch predictions.
- **Responsive Design**: Fully responsive web interface for all devices.
- **Secure & Scalable**: Built with Flask and compliant with industry standards.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Chhavinder058/CAPSTONE.git
   cd CAPSTONE
2. Install the required dependencies:
    pip install -r requirements.txt

3. Ensure the trained model file random_forest_regressor_model.pkl is in the project directory.
    Usage
    Run the Flask application: python appl.py
    Open your browser and navigate to http://127.0.0.1:5000.
    
    Use the web interface to:
    
    Enter crop features manually.
    Upload a CSV file for batch predictions.
   
    Input Features
    Region: North, South, East, West, 
    Soil_Type: Clay, Loam, Sandy, Silt
    Crop: Cotton, Rice, Soybean, Wheat
    Rainfall (mm): Numeric value
    Temperature (°C): Numeric value
    Fertilizer Used: Yes/No
    Irrigation Used: Yes/No
    Weather Condition: Rainy/Sunny
    Days to Harvest: Numeric value
   
    Output
    Predicted crop yield in tons per hectare.
    For batch predictions, a table with predicted yields for each entry.
    
    
Model Details
Algorithm: Random Forest Regressor
Training Framework: PyCaret AutoML
Key Features: Rainfall, Fertilizer, Temperature
Accuracy: R² = 0.91

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For any inquiries, please contact:
Email: W0830731@myscc.ca
GitHub: Chhavinder058
