from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the datasets
accidents = pd.read_csv('accidents.csv')
casualties = pd.read_csv('casualties.csv')
vehicles = pd.read_csv('vehicles.csv')

# Merge datasets to create a training dataset
merged_data = accidents.merge(casualties, on='Accident_Index').merge(vehicles, on='Accident_Index')

# Print column names to inspect
print("Columns in merged_data:")
print(merged_data.columns)

# Select relevant features and target
features = merged_data[['Did_Police_Officer_Attend_Scene_of_Accident', 'Age_of_Driver', 'Vehicle_Type',
                        'Age_of_Vehicle', 'Engine_Capacity_(CC)', 'Day_of_Week', 'Weather_Conditions',
                        'Light_Conditions', 'Road_Surface_Conditions', 'Speed_limit']]
target = merged_data['Accident_Severity']

# Check data types of features
print("Data types of features:")
print(features.dtypes)

# Convert categorical features to numeric
label_encoders = {}
for column in features.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column])
    label_encoders[column] = le

# Verify conversion
print("Data types of features after conversion:")
print(features.dtypes)

# Train a simple model (you should replace this with your trained model)
model = RandomForestClassifier()
model.fit(features, target)

# Save the model
joblib.dump(model, 'model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    police_attend = int(request.form['Did_Police_Officer_Attend'])
    age_of_driver = int(request.form['age_of_driver'])
    vehicle_type = int(request.form['vehicle_type'])
    age_of_vehicle = int(request.form['age_of_vehicle'])
    engine_cc = float(request.form['engine_cc'])
    day_of_week = int(request.form['day'])
    weather_conditions = int(request.form['weather'])
    light_conditions = int(request.form['light'])
    road_surface_conditions = int(request.form['roadsc'])
    speed_limit = int(request.form['speedl'])
    
    # Create input DataFrame with feature names
    input_data = pd.DataFrame([[police_attend, age_of_driver, vehicle_type, age_of_vehicle, engine_cc, 
                                day_of_week, weather_conditions, light_conditions, road_surface_conditions, 
                                speed_limit]],
                              columns=['Did_Police_Officer_Attend_Scene_of_Accident', 'Age_of_Driver', 
                                       'Vehicle_Type', 'Age_of_Vehicle', 'Engine_Capacity_(CC)', 
                                       'Day_of_Week', 'Weather_Conditions', 'Light_Conditions', 
                                       'Road_Surface_Conditions', 'Speed_limit'])
    
    # Load the model
    model = joblib.load('model.pkl')
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    severity = 'slight' if prediction == 3 else 'serious' if prediction == 2 else 'fatal'
    
    return render_template('index.html', output=severity)
@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

if __name__ == '__main__':
    app.run(debug=True)
