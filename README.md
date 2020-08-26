# Lane-Level Traffic speed prediction

This is the code and dataset for the manuscript "Lane-Level Traffic Speed Forecasting: A Novel Mixed Deep Learning Model" which is under review by the Journal of IEEE transaction on Transactions on Intelligent Transportation Systems. 

# File description
DATASET.txt ---The original dataset which was captured by the remote traffic microwave sensors located on the expressways in Beijing. There are eleven properties in this file. The descriptions of the properties are list as follows.

ROAD SECTION, DATE, TIME, L1_VOLUME, L2_VOLUME, L3_VOLUME, L4_VOLUME, L1_SPEED, L2_SPEED, L3_SPEED, L4_SPEED

ROAD SECTION: The ID of the Road section.

DATE: The date of collecting the Traffic flow data.

TIME: The exact time of data acquisition.

L1_VOLUME: The traffic volume of the lane L1 (The inside lane).

L2_VOLUME: The traffic volume of the lane L2.

L3_VOLUME: The traffic volume of the lane L3.

L4_VOLUME: The traffic volume of the lane L4.

L1_SPEED: The traffic speed of the lane L1.

L2_SPEED: The traffic speed of the lane L2.

L3_SPEED: The traffic speed of the lane L3.

L4_SPEED: The traffic speed of the lane L4.

Location of the observation lanes.jpg --- This file introduces the location of the observation lanes.

Structure of the MDL model.jpg --- This file introduces the structures of the MDL model which combines the Conv-LSTM layers, a CNN layer, and a fully connected layer.

MDL_model.py --- This file provides the code of the MDL model.

create_data.py ---This file preprocesses the data which are used as the input of the MDL model.

The pseudo-code of training an MDL model. ---This file provides the pseudo code of the MDL model. 

# Requirements

TensorFlow==1.13.2

Keras==2.1.1

Pandas==0.25.1

Numpy==1.17.2

Matplotlib==3.1.1


# Preprocessing

The MDL employs the datasets which are preprocessed based on the file DATASET.txt, and the speed data and volume data are separated to form two input datasets. The missing and erroneous records were properly remedied by using the historical averaged based data imputation approach. Then, we speed data and volume data were saved as "s.csv" and "v.csv" respectively, which can be fed in the MDL_model.py.

# Python script explanation
create_data.py:

create_dataset_input(data,look_back,look_ahead)----create the dataset for input

create_dataset_output(data,look_back,look_ahead)---create the dataset for output

More description are shown in MDL_model.py.
