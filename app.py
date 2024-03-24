import streamlit as st
from joblib import load
import sklearn
import time
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import datetime
import joblib

# Load the transformer and model
transformer = joblib.load('transformer.joblib')
model = joblib.load('decision_tree_model.joblib')

st.header('Austin Bike Risk Assessment',divider='grey')
st.subheader('Crash severity',)

shadow = "0px 0px 10px rgba(209,211,224,1.00)"  # Shadow effect


circle_style = """
 <div style='display: flex; justify-content: space-around; align-items: center; margin: 20px 0;'>
     <div style='width: 100px; height: 100px; border-radius: 50%; background-color: red; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold;'>High</div>
     <div style='width: 100px; height: 100px; border-radius: 50%; background-color: orange; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold;'>Medium</div>
     <div style='width: 100px; height: 100px; border-radius: 50%; background-color: green; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold;'>Low</div>
 </div>
 """

st.markdown(circle_style, unsafe_allow_html=True)

st.header('',divider='grey')

st.sidebar.header('Calculate Your Risk of Riding a Bike in Austin')
day_of_week = st.sidebar.selectbox("Day of Week",index =None,options=["Monday", "Tuesday", "Wednesday", "Thursday","Friday","Saturday","Sunday"])

ride_time = st.sidebar.time_input("Time of Ride",value= None)

Roadway_Part = st.sidebar.selectbox("Select Roadway Part",index =None,options=["Main/Proper Lane", "Service/Frontage Road", "Entrance/On Ramp", "Other"])

Speed_Limit = st.sidebar.number_input("Speed Limit", min_value=0, step=1)

Surface_Condition = st.sidebar.selectbox("Select Surface Condition",index =None,options=["Dry", "Wet", "Other"])

Person_Helmet = st.sidebar.selectbox("Helmet worn?",index =None,options=["Not Worn", "Worn"])


# Assuming the prediction part remains the same
if st.sidebar.button('Predict Risk'):
    with st.status("Processing...", expanded=True) as status:
        st.write("Preparing input data...")
        time.sleep(1)  # Simulating time taken to prepare data
        

        # Preparation of input data
        time_of_ride_formatted = f"{ride_time.hour:02d}{ride_time.minute:02d}"
        input_data = pd.DataFrame({
            'Day of Week': [day_of_week],
            'Crash Time': [time_of_ride_formatted],
            'Roadway Part': [Roadway_Part],
            'Speed Limit': [Speed_Limit],
            'Surface Condition': [Surface_Condition],
            'Person Helmet': [Person_Helmet]
        }, index=[0])

        st.write("Making prediction...")
        time.sleep(2)  # Simulating time taken for prediction

        # Transforming input and making prediction
        new_input_transformed = transformer.transform(input_data)
        prediction = model.predict(new_input_transformed)

        # Update status to complete
        status.update(label="Prediction complete!", state="complete", expanded=False)

    # Determine the color and text based on the prediction
    if prediction[0] == "Low":
        color = "green"
        risk_text = "Low Risk"
    elif prediction[0] == "Medium":
        color = "orange"
        risk_text = "Medium Risk"
    else:  # Assuming 'High'
        color = "red"
        risk_text = "High Risk"

    descriptive_labels = {
    'Low': " Risk of injury is low 😊. Enjoy your ride, but always stay safe on the road.🚴",
    'Medium': "There is a moderate risk of injury ⚠️. Stay alert and adhere to safety measures.🛑",
    'High': "There is a high risk of severe injury in case of a crash 🚨. Please exercise extreme caution.🚑"
}
    description = descriptive_labels[prediction[0]]

    risk_advice = {
    'Low': "😌 Continue to follow traffic rules, wear a helmet 🚴‍♂️, and remain vigilant to maintain low risk.",
    'Medium': "🤔 Consider avoiding high traffic areas, wearing reflective clothing 🦺, and using bike lights 🔦.",
    'High': "🚨 It is advisable to avoid riding in these conditions if possible. If you must ride, wear protective gear 🛡️ and plan a route with less traffic and lower speed limits 🛣️."
}
    advice = risk_advice[prediction[0]]
 
    st.markdown(f"""
    <div style='display: flex; align-items: center; justify-content: start; gap: 20px;'>
    <div style='flex-shrink: 0; width: 150px; height: 150px; border-radius: 50%; background-color: {color}; display: flex; align-items: center; justify-content: center; font-weight: bold; box-shadow: {shadow};'>
        <span style='color: {color}; font-size: 38px; text-align = center;'>{risk_text}</span>
    </div>
    <div>
        <p><b> Description:</b></p>
        <div style='text-align: left;'>
            <p>{description}</p>
        </div>
        <p><b> Actionable Insights:</b></p>
        <div style='text-align: left;'>
            <p>{advice}</p>
        </div>
    </div>
    </div>

# """, unsafe_allow_html=True)





