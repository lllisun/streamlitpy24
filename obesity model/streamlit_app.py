import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("rf_obesity.pkl", "rb") as f:
    model = pickle.load(f)

# App title
st.title("Obesity Level Prediction")

# User inputs for prediction
st.header("Enter Your Information")

# Age (Integer)
age = st.number_input("Age (years)", min_value=1, max_value=120, value=25)

# Height in feet and inches (converted to meters for model)
height_feet = st.number_input("Height (feet)", min_value=1, max_value=8, value=5)
height_inches = st.number_input("Additional Height (inches)", min_value=0, max_value=11, value=7)
height_meters = (height_feet * 12 + height_inches) * 0.0254  # Convert to meters

# Weight in pounds (converted to kilograms for model)
weight_pounds = st.number_input("Weight (pounds)", min_value=20, max_value=500, value=150)
weight_kg = weight_pounds * 0.453592  # Convert to kilograms

# Other features
fh = st.selectbox("Family History of Overweight (FH)", options=["yes", "no"])
favc = st.selectbox("Frequent Consumption of High-Calorie Food (FAVC)", options=["yes", "no"])
fcvc = st.slider("Frequency of Vegetable Consumption (1=Low, 3=High)", min_value=1, max_value=3, value=2)
ncp = st.slider("Number of Meals per Day (NCP)", min_value=1, max_value=4, value=3)
caec = st.selectbox("Consumption of Food Between Meals (CAEC)", options=["Always", "Frequently", "Sometimes", "Never"])
smoke = st.selectbox("Do You Smoke? (SMOKE)", options=["yes", "no"])
ch2o = st.slider("Daily Water Intake (liters, CH2O)", min_value=1.0, max_value=3.0, step=0.1)
scc = st.selectbox("Do You Monitor Calories? (SCC)", options=["yes", "no"])
faf = st.slider("Physical Activity Frequency (days/week, FAF)", min_value=0.0, max_value=7.0, step=0.1)
tue = st.slider("Time Spent Using Technology (hours/day, TUE)", min_value=0.0, max_value=24.0, step=0.5)
calc = st.selectbox("Consumption of Alcohol (CALC)", options=["no", "sometimes", "frequently"])
mtrans = st.selectbox("Mode of Transportation (MTRANS)", options=["Walking", "Public_Transportation", "Automobile", "Bike", "Motorbike"])

# Prepare the input data for prediction
input_data = pd.DataFrame({
    "Age": [age],
    "Height": [height_meters],
    "Weight": [weight_kg],
    "FH_yes": [1 if fh == "yes" else 0],
    "FAVC_yes": [1 if favc == "yes" else 0],
    "FCVC": [fcvc],
    "NCP": [ncp],
    "CAEC_Always": [1 if caec == "Always" else 0],
    "CAEC_Frequently": [1 if caec == "Frequently" else 0],
    "CAEC_Sometimes": [1 if caec == "Sometimes" else 0],
    "SMOKE_yes": [1 if smoke == "yes" else 0],
    "CH2O": [ch2o],
    "SCC_yes": [1 if scc == "yes" else 0],
    "FAF": [faf],
    "TUE": [tue],
    "CALC_no": [1 if calc == "no" else 0],
    "CALC_sometimes": [1 if calc == "sometimes" else 0],
    "CALC_frequently": [1 if calc == "frequently" else 0],
    "MTRANS_Walking": [1 if mtrans == "Walking" else 0],
    "MTRANS_Public_Transportation": [1 if mtrans == "Public_Transportation" else 0],
    "MTRANS_Automobile": [1 if mtrans == "Automobile" else 0],
    "MTRANS_Bike": [1 if mtrans == "Bike" else 0],
    "MTRANS_Motorbike": [1 if mtrans == "Motorbike" else 0],
})

# Align input data to the model's training columns
for col in model.feature_names_in_:  # Adjust this line if the model uses `X.columns` or similar
    if col not in input_data.columns:
        input_data[col] = 0

# Class labels for one-hot encoded predictions
class_labels = [
    "Normal_Weight",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
    "Overweight_Level_I",
    "Overweight_Level_II",
]

if st.button("Predict"):
    # Reorder columns to match the training set
    input_data = input_data[model.feature_names_in_]
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Find the predicted class
    predicted_class_index = prediction[0].argmax()  # Index of the highest value
    predicted_class_label = class_labels[predicted_class_index]
    
    # Display the result
    st.write(f"Predicted Obesity Level: {predicted_class_label}")
