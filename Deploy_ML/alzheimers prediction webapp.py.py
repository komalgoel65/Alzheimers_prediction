# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 08:26:45 2024

@author: ADMIN
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/ADMIN/OneDrive/Documents/Desktop/Alzheimers_prediction model/trained model.sav','rb'))

with open('C:/Users/ADMIN/OneDrive/Documents/Desktop/Alzheimers_prediction model/scaler_.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)
    
# creating a function for prediction

def Alzheimers_prediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the input data
    std_data = loaded_scaler.transform(input_data_reshaped)
    print(std_data)

    prediction = loaded_model.predict(std_data)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is healthy'
    else:
      return 'The person is having alzheimers'

def main():
    #giving a title
    st.title('Alzheimers prediction web app')
    
    #getting the input data from the user
    #Age,Gender,BMI,Smoking,AlcoholConsumption,PhysicalActivity,DietQuality,SleepQuality,FamilyHistoryAlzheimers,CardiovascularDisease,Diabetes,Depression,HeadInjury,Hypertension,SystolicBP,DiastolicBP,CholesterolTotal,MMSE,FunctionalAssessment,MemoryComplaints,BehavioralProblems,Disorientation,PersonalityChanges,DifficultyCompletingTasks,Forgetfulness
    Age = st.text_input('Age of the person')
    Gender = st.text_input('Gender of the person')
    BMI = st.text_input('BMI value')
    Smoking = st.text_input('smoke or not')
    AlcoholConsumption = st.text_input('Consumes Alcohol or not')
    PhysicalActivity = st.text_input('Physical activity')
    DietQuality = st.text_input('Diet Quality')
    SleepQuality = st.text_input('Sleep Quality')
    FamilyHistoryAlzheimers = st.text_input('Any family History of Alzheimers (0/1)')
    CardiovascularDisease = st.text_input('Cardiovascular Diseases(0/1)')
    Diabetes = st.text_input('Diabetic (0/1)')
    Depression = st.text_input('Depression (0/1)')
    HeadInjury = st.text_input('Head Injury (0/1)')
    Hypertension = st.text_input('Hypertension (0/1)')
    SystolicBP = st.text_input('SystolicBP value')
    DiastolicBP = st.text_input('DiastolicBP value')
    CholesterolTotal = st.text_input('Total Cholestrol value')
    MMSE = st.text_input('MMSE value')
    FunctionalAssessment = st.text_input('Functional Assessment')
    MemoryComplaints = st.text_input('Memory Complaints (0/1)')
    BehavioralProblems = st.text_input('Behavioural Problems (0/1)')
    Disorientation = st.text_input('Disorientation (0/1)')
    PersonalityChanges = st.text_input('Personality changes (0/1)')
    DifficultyCompletingTasks = st.text_input('Difficulty completing tasks (0/1)')
    Forgetfulness = st.text_input('Forgetfullness (0/1)')
    
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prdiction
    
    
    if st.button('Alzheimers test Results'):
        diagnosis = Alzheimers_prediction([Age,Gender,BMI,Smoking,AlcoholConsumption,PhysicalActivity,DietQuality,SleepQuality,FamilyHistoryAlzheimers,CardiovascularDisease,Diabetes,Depression,HeadInjury,Hypertension,SystolicBP,DiastolicBP,CholesterolTotal,MMSE,FunctionalAssessment,MemoryComplaints,BehavioralProblems,Disorientation,PersonalityChanges,DifficultyCompletingTasks,Forgetfulness])
    
    st.success(diagnosis)
if __name__=='__main__':
    main()    
    
        