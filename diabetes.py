import streamlit as st
import numpy as np
import pickle

# Load model and scaler 
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title('🩺 Diabetes Prediction App')

st.markdown("Enter the patient's information below:")

# Input fields 
pregnancies = st.number_input('Pregnancies', min_value=0)
glucose = st.number_input('Glucose', min_value=0)
blood_pressure = st.number_input('Blood Pressure', min_value=0)
skin_thickness = st.number_input('Skin Thickness', min_value=0)
insulin = st.number_input('Insulin', min_value=0)
bmi = st.number_input('BMI', min_value=0.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0)
age = st.number_input('Age', min_value=0)
 

if st.button('Predict'):
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]).reshape(1, -1)
    std_input = scaler.transform(input_data)
    prediction = model.predict(std_input)

    if prediction[0] == 0:
        st.success(' The person is **not diabetic**.')
    else:
        st.error('The person **is diabetic**.')
