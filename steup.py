import pandas as pd
import joblib

loaded_model = joblib.load('weather_prediction_model.pkl')


hum = input("Enter the Humidity : ")
pre = input("Enter Pressure : ")
prc = input("Enter the Precipitation : ")

# Sample new data for prediction
new_data = pd.DataFrame({
    'BASEL_humidity': [hum],  
    'BASEL_pressure': [pre],  
    'BASEL_precipitation': [prc],  
    'Month': [7], 
    'Year': [2023],  
})

# new_data = pd.DataFrame({
#     'BASEL_humidity': [0.73],  
#     'BASEL_pressure': [1.008],  
#     'BASEL_precipitation': [9.3],  
#     'Month': [7], 
#     'Year': [2023],  
# })

# Use the loaded model to make predictions on the new data
predictions = loaded_model.predict(new_data)
print(predictions)