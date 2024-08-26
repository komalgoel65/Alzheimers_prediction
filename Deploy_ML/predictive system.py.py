# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

import pickle

# loading the saved model
loaded_model = pickle.load(open('C:/Users/ADMIN/OneDrive/Documents/Desktop/Alzheimers_prediction model/trained model.sav','rb'))

with open('C:/Users/ADMIN/OneDrive/Documents/Desktop/Alzheimers_prediction model/scaler_.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)

input_data = (82,1,36.22309879636211,0,4.19289550965694,6.381502407159495,7.971127068871205,9.521026998926784,0,1,0,0,0,0,120,93,271.84144940568734,8.984758990788695,6.946428266027044,0,1,0,0,0,1)

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
  print('The person is healthy')
else:
  print('The person is having alzheimers')