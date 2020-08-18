# Lane-Level_Traffic_speed_prediction

This is the code and dataset for the manuscript "Lane-Level Traffic Speed Forecasting: A Novel Mixed Deep Learning Model" which is under review by the Journal of IEEE transcation on Transactions on Intelligent Transportation Systems.


# File description
DATASET.txt ---The orginall dataset which were captured by the remote traffic microwave sensors located on the expressways in Beijing. There are eleven properties in this file.

ROAD SECTION: The ID of the Road section.

DATE:The data of collecting the Traffic flow data.

TIME: The exact time of data acquisition.

L1_VOLUME: The traffic volume of the Lane 1 (The inside lane).

L2_VOLUME: The traffic volume of the Lane 2.

L3_VOLUME: The traffic volume of the Lane 3.

L4_VOLUME: The traffic volume of the Lane 4.

L1_SPEED: The traffic speed of the Lane 1.

L2_SPEED: The traffic speed of the Lane 2.

L3_SPEED: The traffic speed of the Lane 3.

L4_SPEED: The traffic speed of the Lane 4.

Location of the observation lanes.jpg --- This file introduces the location of the observation lanes.

Structure of the MDL model.jpg --- This file introduces the structures of the MDL model which combines the Conv-LSTM layers, a CNN layer, and a fully connected layer.

