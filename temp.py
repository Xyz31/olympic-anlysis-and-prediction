import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset
df = pd.read_csv('C:/Users/Kashaf/Desktop/Vedio/Athletes/processed_data.csv')

# Drop irrelevant columns
df = df[['Sex', 'Age', 'Height', 'Weight', 'region', 'Sport', 'Medal']]
df = df.dropna()

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['Sex', 'region', 'Sport'])

# Split data into features and target variable
X = df_encoded.drop('Medal', axis=1)
y = df_encoded['Medal']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Input parameters for prediction
sex = st.selectbox("Select Sex", ["M", "F"])
age = st.slider("Select Age", 10, 97)
height = st.slider("Select Height (In centimeters)", 127, 226)
weight = st.slider("Select Weight (In kilograms)", 25, 214)
region = st.selectbox("Select Country", df['region'].unique())
sport = st.selectbox("Select Sport", df['Sport'].unique())

# Collect unique sport names from the original data
unique_sports = df['Sport'].unique()

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Sex_F': [1 if sex == 'F' else 0],
    'Sex_M': [1 if sex == 'M' else 0],
    'Age': [age],
    'Height': [height],
    'Weight': [weight],
    'region_' + region: [1]
})

for sport in unique_sports:
    input_data['Sport_' + sport] = [1 if sport == sport else 0]

# Predict the probability of winning a medal
probability = model.predict_proba(input_data)[0, 1]

# Display the predicted probability
st.write(f"The predicted probability of winning a medal is: {probability}")