import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
weather_data = pd.read_csv('weather_prediction_dataset.csv')
temp_weather_data = pd.read_csv('weather_prediction_dataset.csv')

# Check the structure, missing values, and types
# print(weather_data.head())
# print(weather_data.info())
# print(weather_data.isnull().sum())

# Fill missing values with the mean
weather_data.fillna(weather_data.mean(), inplace=True)

# Convert date columns to datetime
weather_data['DATE'] = pd.to_datetime(weather_data['DATE'], format='%Y%m%d')
temp_weather_data['DATE'] = pd.to_datetime(weather_data['DATE'], format='%Y%m%d')


# Step 3: Data Visualization and Exploration
# Line plot to visualize the temperature mean over time in Basel
# plt.figure(figsize=(10, 6))
# plt.plot(weather_data['DATE'], weather_data['BASEL_temp_mean'], label='Basel Temp Mean')
# plt.title('Temperature Mean in Basel Over Time')
# plt.xlabel('Date')
# plt.ylabel('Temperature (°C)')
# plt.legend()
# plt.show()


# Correlation heatmap to understand feature relationships
# correlation_matrix = weather_data.corr()
# plt.figure(figsize=(12, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.show()



selected_features = ['BASEL_temp_mean', 'BASEL_humidity', 'BASEL_pressure', 'BASEL_precipitation']
weather_data = weather_data[selected_features]

# Feature engineering: Extracting month and year from the date
weather_data['Month'] = temp_weather_data['DATE'].dt.month
weather_data['Year'] = temp_weather_data['DATE'].dt.year

# Drop the date column as we've extracted year and month
# weather_data.drop('DATE', axis=1, inplace=True)

# Display the updated dataframe after feature engineering
print(weather_data.head())

from sklearn.model_selection import train_test_split

# Define features and target
X = weather_data.drop('BASEL_temp_mean', axis=1)  # Features
y = weather_data['BASEL_temp_mean']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Visualize predicted vs. actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()

import joblib

# Assuming 'model' is your trained model
joblib.dump(model, 'weather_prediction_model.pkl')


