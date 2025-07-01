import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder

model=tf.keras.models.load_model('model.h5')
with open('encoder_gender.pkl','rb') as file:
    encoder_gender=pickle.load(file)

with open('encoder_geography.pkl','rb') as file:
    encoder_geography=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file) 

st.title('Customer Churn PRediction')
  

geography = st.selectbox('Geography', encoder_geography.classes_)
gender = st.selectbox('Gender', encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

encoded_geo = encoder_geography.transform([geography])

encoded_geo_df = pd.DataFrame(encoded_geo, columns=['Geography'])
input_data = pd.concat([input_data.reset_index(drop=True), encoded_geo_df], axis=1)
column_order = scaler.feature_names_in_ 
input_data = input_data[column_order]
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
