# Lane-Level Traffic speed prediction

This is the code for the manuscript "Lane-Level Traffic Speed Forecasting: A Novel Mixed Deep Learning Model" 

# File description
MDL_model.py --- This file provides the code of the MDL model.
create_data.py ---This file preprocesses the data which are used as the input of the MDL model.
The pseudo-code of training an MDL model. ---This file provides the pseudo code of the MDL model. 

# Requirements

TensorFlow==1.13.2
Keras==2.1.1
Pandas==0.25.1
Numpy==1.17.2
Matplotlib==3.1.1


# Python script explanation

create_data.py---
create_dataset_input(data,look_back,look_ahead)----create the dataset for input
create_dataset_output(data,look_back,look_ahead)---create the dataset for output
