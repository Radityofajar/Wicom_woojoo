# Wicom_woojoo

### Data
Contains all the datasets that collected from sensor located in WicomAI LAB, Kookmin University

### model
Contains initial models (temperature, humidity, waterlevel, waterleak, door, fire)

### postproc.py
This is the main program to run the anomaly detection using postprocess features by IoT.own Coxlab.
Step 1: Prepare the initial model using model_init.py
Step 2: open postproc.py and change parameter as needed
-> Thresholding value parameter: to create upper and lower boundaries.
-> sliding window parameter: will be explain later
-> model initialization parameter: to use what kind of model needed and also how you want to retrain it.

#### Sliding window
Parameter: batch_size and train_number
![sliding_window](/docs/sliding_window.png)
