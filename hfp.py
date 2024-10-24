import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open('a.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.markdown("<h1 style='text-align: center;'>Heart Failure Detection</h1>", unsafe_allow_html=True)


# Create two columns
# Adjust the columns width to move the left column further left and the right column further right
col1, col2 = st.columns([1, 3.9])  # Adjust the ratios as needed
 # 30% left, 70% right

# Left column for the image and buttons
with col1:
    st.image("https://static.vecteezy.com/system/resources/thumbnails/021/360/193/small_2x/doctor-character-illustration-free-png.png", width=100)
    st.markdown("<h3>Navigation</h3>", unsafe_allow_html=True)

    # Check if page is in session state, if not, initialize it
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'  # Default page is 'home'

    # Check if user_name is in session state, if not, initialize it
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""  # Initialize with empty string

    if st.button("Home"):
        st.session_state.page = "Home"
    if st.button("Model-Prediction"):
        st.session_state.page = 'Model-Prediction'
    if st.button("Prediction"):
        st.session_state.page = 'prediction'
    if st.button("Dataset"):
        st.session_state.page = 'dataset'
    if st.button("About"):
        st.session_state.page = 'about'

# Function to display user input and make predictions
def display_prediction(input_data):
    sex_map = {"Male": 1, "Female": 0}
    fasting_bs_map = {"Yes": 1, "No": 0}
    exercise_angina_map = {"Yes": 1, "No": 0}
    chest_pain_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
    resting_ecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    st_slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

    input_vector = np.array([
        input_data['age'],
        sex_map[input_data['sex']],
        chest_pain_map[input_data['chest_pain_type']],
        input_data['resting_bp'],
        input_data['cholesterol'],
        fasting_bs_map[input_data['fasting_bs']],
        resting_ecg_map[input_data['resting_ecg']],
        input_data['max_hr'],
        exercise_angina_map[input_data['exercise_angina']],
        input_data['oldpeak'],
        st_slope_map[input_data['st_slope']]
    ]).reshape(1, -1)

    prediction = model.predict(input_vector)

    st.subheader("Your Input Data:")
    st.markdown(f"""
    Age: {input_data['age']}  
    Sex: {input_data['sex']}  
    Chest Pain Type: {input_data['chest_pain_type']}  
    Resting Blood Pressure: {input_data['resting_bp']} mm Hg  
    Cholesterol: {input_data['cholesterol']} mg/dl  
    Fasting Blood Sugar > 120 mg/dl: {input_data['fasting_bs']}  
    Resting ECG: {input_data['resting_ecg']}  
    Maximum Heart Rate Achieved: {input_data['max_hr']} bpm  
    Exercise-Induced Angina: {input_data['exercise_angina']}  
    Oldpeak (ST Depression): {input_data['oldpeak']}  
    ST Slope: {input_data['st_slope']}
    """)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.markdown(f'<div style="border: 2px solid red; padding: 10px;"><h4 style="color:red;">⚠ Warning: {st.session_state.user_name}, the model predicts a high risk of heart failure.</h4></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="border: 2px solid green; padding: 10px;"><h4 style="color:green;">✅ Good news: {st.session_state.user_name}, the model predicts a low risk of heart failure.</h4></div>', unsafe_allow_html=True)

# Right column for dynamic content
with col2:
    if st.session_state.page == 'Model-Prediction':
        st.markdown('<h2>Welcome to the Heart Failure Detection app!</h2>', unsafe_allow_html=True)
        st.write("Use the inputs to predict the risk of heart failure.")

        # User input
        user_name = st.text_input("Please enter your name")

        # Input fields
        age = st.number_input("Age", min_value=5, max_value=120, value=None)
        sex = st.selectbox("Sex", options=["", "Male", "Female"], index=0)
        chest_pain_type = st.selectbox("Chest Pain Type", options=["", "Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"], index=0)
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=None)
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=400, value=None)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["", "Yes", "No"], index=0)
        resting_ecg = st.selectbox("Resting ECG", options=["", "Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"], index=0)
        max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=None)
        exercise_angina = st.selectbox("Exercise-Induced Angina", options=["", "Yes", "No"], index=0)
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, step=0.1, value=None)
        st_slope = st.selectbox("ST Slope", options=["", "Upsloping", "Flat", "Downsloping"], index=0)

        # Store name in session state
        if st.button("Predict"):
            if user_name:
                st.session_state.user_name = user_name
            else:
                st.error("Please enter your name.")

            if (resting_bp is not None and resting_bp >= 50 and
                cholesterol is not None and cholesterol >= 100 and
                max_hr is not None and max_hr >= 60 and
                oldpeak is not None and oldpeak >= 0.0):
                st.session_state.page = 'prediction'
                st.session_state.input_data = {
                    'age': age,
                    'sex': sex,
                    'chest_pain_type': chest_pain_type,
                    'resting_bp': resting_bp,
                    'cholesterol': cholesterol,
                    'fasting_bs': fasting_bs,
                    'resting_ecg': resting_ecg,
                    'max_hr': max_hr,
                    'exercise_angina': exercise_angina,
                    'oldpeak': oldpeak,
                    'st_slope': st_slope
                }
            else:
                st.error("Please provide valid inputs for all fields.")

    elif st.session_state.page == 'prediction':
        st.markdown('<h2>Prediction Results</h2>', unsafe_allow_html=True)
        if 'input_data' not in st.session_state:
            st.error("No input data found! Please go to the Home page and provide the necessary inputs.")
        else:
            display_prediction(st.session_state.input_data)

    elif st.session_state.page == 'about':
        st.subheader("About This Project")
        st.write("""
            This Heart Failure Detection application leverages machine learning 
            to predict heart failure risk based on various health parameters. 
            Enter details to receive insights into your heart health. This tool is 
            for educational purposes and should not replace medical advice.
        """)

    elif st.session_state.page == 'dataset':
        st.subheader("Dataset Information")
        st.write("""
            The dataset includes health attributes such as age, resting BP, cholesterol, 
            etc., that are used for training the model. The model has been optimized 
            for predicting heart failure risk accurately.
        """)
    

        data=pd.read_csv('heart.csv')
        x=data.head()
        st.write(x)
    elif st.session_state.page == 'Home':
         st.header("The Heart Disease")

         st.write("""A heart attack, or myocardial infarction, occurs when a section of the heart muscle is deprived of oxygen-rich blood, leading to potential damage. In India, coronary artery disease (CAD) is the primary culprit, often stemming from lifestyle factors such as poor diet, lack of exercise, and increasing stress levels. 

          The significance of timely treatment cannot be overstated; every moment counts in restoring blood flow to minimize damage to the heart. Additionally, while CAD is the leading cause, there are instances where severe spasms of the coronary arteries can also halt blood flow, although this is less common.

          In India, awareness around heart health is crucial, especially given the rise in risk factors like diabetes, hypertension, and obesity. Promoting a balanced diet, regular physical activity, and stress management can significantly help in preventing heart attacks. Community health initiatives and regular health check-ups can play an important role in early detection and intervention..""")

         st.image("ty.jpg")
         st.subheader("Symptoms")

         st.write("""
                      The major symptoms of a heart attack are

          - Chest pain or discomfort. Most heart attacks involve discomfort in the center or left side of the chest that lasts for more than a few minutes or that goes away and comes back. The discomfort can feel like uncomfortable pressure, squeezing, fullness, or pain.
          - Feeling weak, light-headed, or faint. You may also break out into a cold sweat.
          - Pain or discomfort in the jaw, neck, or back.
          - Pain or discomfort in one or both arms or shoulders.
          - Shortness of breath. This often comes along with chest discomfort, but shortness of breath also can happen before chest discomfort.
          """)

         st.subheader("Risk factors")

         st.write("""Several health conditions, your lifestyle, and your age and family history can increase your risk for heart disease and heart attack. These are called risk factors. About half of all Americans have at least one of the three key risk factors for heart disease: high blood pressure, high blood cholesterol, and smoking.

          Some risk factors cannot be controlled, such as your age or family history. But you can take steps to lower your risk by changing the factors you can control.
          """)

         st.subheader("Recover after a heart attack")

         st.write("""
                  If you’ve had a heart attack, your heart may be damaged. This could affect your heart’s rhythm and its ability to pump blood to the rest of the body. You may also be at risk for another heart attack or conditions such as stroke, kidney disorders, and peripheral arterial disease (PAD).

          You can lower your chances of having future health problems following a heart attack with these steps:

          - Physical activity—Talk with your health care team about the things you do each day in your life and work. Your doctor may want you to limit work, travel, or sexual activity for some time after a heart attack.
          - Lifestyle changes—Eating a healthier diet, increasing physical activity, quitting smoking, and managing stress—in addition to taking prescribed medicines—can help improve your heart health and quality of life. Ask your health care team about attending a program called cardiac rehabilitation to help you make these lifestyle changes.
          - Cardiac rehabilitation—Cardiac rehabilitation is an important program for anyone recovering from a heart attack, heart failure, or other heart problem that required surgery or medical care. Cardiac rehab is a supervised program that includes
          1. Physical activity
          2. Education about healthy living, including healthy eating, taking medicine as prescribed, and ways to help you quit smoking
          3. Counseling to find ways to relieve stress and improve mental health

          A team of people may help you through cardiac rehab, including your health care team, exercise and nutrition specialists, physical therapists, and counselors or mental health professionals.


          """)
                