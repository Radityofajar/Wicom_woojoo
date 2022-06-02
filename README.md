# Wicom_woojoo

### Data
Contains all the datasets that collected from sensor located in WicomAI LAB, Kookmin University

### model
Contains initial models (temperature, humidity, waterlevel, waterleak, door, fire)

### postproc.py
This is the main program to run the anomaly detection based on Isolation Forest using postprocess features by IoT.own Coxlab.
(nb: flow_postproc.py is similar with postproc.py but more readable and easily modify)

  Step 1: Prepare the initial model using model_init.py
  
  Step 2: open postproc.py and change parameter as needed
  
    -> Thresholding value parameter: to create upper and lower boundaries.
    
    -> sliding window parameter: to keep the uniqueness and rarity.
    
    -> model initialization parameter: to use what kind of model needed and also how you want to retrain it.
  Step 3: Run the postproc.py program

#### Sliding window
Parameter: batch_size and train_number
![sliding_window](/docs/sliding_window.png)

### postproc_HBOS.py
Histogram- based outlier detection (HBOS) is an efficient unsupervised method. It assumes the feature independence and calculates the degree of outlyingness by building histograms.

### postproc_MCD.py
Outlier Detection with Minimum Covariance Determinant (MCD)
