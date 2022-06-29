# Wicom_woojoo (Description is not finish)

### System Requirements
    Python 3
    Pyiotown >> python3 -m pip install pyiotown
    sklearn >>   python3 -m pip install scikit-learn
    PyOD >>  python3 -m pip install pyod    
    joblib >>  python3 -m pip install joblib
    numpy >>  python3 -m pip install numpy
    pandas >>  python3 -m pip install pandas 
### Data
It contains all the datasets collected from sensors located in WicomAI LAB, Kookmin University
### model
It contains initial models (temperature, humidity, waterlevel, waterleak, door, and fire)
### Main Program:
    1. wicom_postproc_fire.py
    2. wicom_postproc_tdhd.py
    3. wicom_postproc_thtd.py
    4. wicom_postproc_wlak.py
    5. wicom_postproc_wlvl.py
We distinguish the post-process programs based on the sensor type.
#### 1. Fire detector (*fire*)
    {'dtype' : 'fire', 'nid': 'WS000001FFFF123456','val0' : 0, 'val1' : 25.7}
    Meaning of val0 value: fire state, type=int, possible value:0=no fire, 1=fire.
    Meaning of val1 value: temperature value, type = one decimal place float.

#### 2. Temperature, humidity, and door measuring instruments (*tdhd*)
    {'dtype': 'tdhd', 'nid': 'WS000001FFFF123456', 'val0': 28.3, 'val1': 48.2, 'val2': 1}
    Meaning of val0 value: temperature value, type = one decimal place float.
    Meaning of val1 value: Humidity value, type = one decimal place float.
    Meaning of val2 value: opening/closing state, type=int, possible value:0=door closed, 1=door open

#### 3. Temperature and humidity measuring instruments (*thtd*)
    {'dtype': 'thtd', 'nid': 'WS000001FFFF123456', 'val0': 27.8, 'val1': 56.2}
    Meaning of val0 value: temperature value, type = one decimal place float.
    Meaning of val1 value: Humidity value, type = one decimal place float.

#### 4. Water leakage sensor (*wlak*)
    {"dtype":"wlak", "nid":"WS000001FFFF123456", "val0":0}
    Meaning of val0 value: water leakage state, type=int, possible value:0=no leakage, 1=leakage.
    
#### 5. Water level measuring instruments (*wlvl*)
    {"dtype":"wlvl", "nid":"WS000001FFFF123456","val0":719}
    Meaning of val0 value: water level value, type=int, unit=mm
    
### Data Input Format
Data forms related to the Qubics IoT sensor module. We receive data from the Qubics IoT sensor module every 1 minute and send it directly to post-process program. The data input format is shown below.
![data_input_format](/docs/data_input_tdhd.JPG)
*data_acquisition.py* is a data acquisition program to collect data from the Qubics IoT sensor module and post it to the IoT.own server.

If the data input format is different, please configure the **receive_data function (def receive_data(rawdata))** inside the post-process program. The expected output of this function is : 

{"dtype":"wlvl", "nid":"WS000001FFFF123456","val0":719} or 

{'dtype': 'tdhd', 'nid': 'WS000001FFFF123456', 'val0': 28.3, 'val1': 48.2, 'val2': 1}

### Data Output Format
The output from post-process program will follow the format below:

![data-output_format](/docs/data_output_tdhd.JPG)

You can modify the data output format by configuring the *changedata* part inside the post-process program.

        For example:
        changedata = {}
        changedata['dtype'] = message['dtype']
        changedata['nid'] = message['nid']
        changedata['val0'] = float(message['val0'])
        changedata['result_v0'] = sensor_wlvl_status
        changedata['anomaly_score'] = float(anomaly_score_wlvl)
        changedata['anomaly_score_threshold'] = float(anomaly_threshVal0)
        rawdata['data'] = changedata
        return rawdata

### Parameter Explanation
We can change some parameters to set the sensitivity of the anomaly detection program. For example, If the program has many false alarms, we can reduce the sensitivity and vice versa.
There are several parameters that we can tune to adjust the sensitivity:

#### 1. Outlier_fraction
This parameter sets the amount of contamination of the data set (the proportion of the outliers in the data set). They are used when fitting/training to define the threshold on the scores of the samples.

Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

        Recommendation value: 0.025 - 0.03 (or 2.5% - 3%)

#### 2. Anomaly_score_threshold
This parameter is for calculating the baseline to determine whether the data anomaly score is an outlier or inlier.

![anomaly_score_threshold](/docs/anomaly_score_threshold.png)

        Anomaly score threshold = (mean - (standard deviation * x))
        where: x =  user input to tuning the anomaly score threshold
#### 3. Boundaries threshold
Parameter: *upper_threshold* and *lower_treshold*

These parameters are used for the simple thresholds to detect extreme values.

![boundaries_threshold](/docs/boundaries_threshold.png)

#### 4. Sliding window
Parameter: *batch_size* and *train_number*

These parameters are used for the re-training process (online training). It determines the number of data used in training the Isolation forest model.

![sliding_window](/docs/sliding_window.png)

## How to run
There are several parameters that have to fill before run the program.

py wicom_postprocv2_wlvl.py **[URL] [name] [token] [low_threshVal0] [up_thresVal0] [batchsize] [train_number] [outlier_fraction] [anomaly_threshVal0]**

    For example:
    [URL] = https://town.coxlab.kr/
    [name] = username@gmail.com
    [token] = c34859e08fa526f64288182
    lower_threshold = 0
    upper_threshold = 3500
    batchsize = 60 
    train_number = 1440
    outlier_fraction = 0.03
    anomaly_score_threshold = 2.85
py wicom_postproc_wlvl.py **https://town.coxlab.kr/ username@gmail.com c34859e08fa526f64288182 250 3500 60 1440 0.03 2.85**

## Conclusions
•	Regarding training model time and prediction time, the Isolation Forest algorithm is fast. Make it suitable for real-time streaming data.

•	We can change the parameters to adjust the sensitivity, but if the anomaly detection sensitivity is too high, it will outcome many false alarms. 

•	In contrast, if the sensitivity is too low, then it will outcome many missed alarms. 

•	So, we must find the right tune to adjust the best sensitivity for each sensor.

## Additional

### Comparison
![model_comparison1](/docs/result_timecomplexity.JPG)
![model_comparison2](/docs/result_roc.JPG)
![model_comparison3](/docs/result_precision.JPG)

The tables above show the comparison between each anomaly detection algorithm. We choose the best 3 among them (Isolation Forest, HBOS, and MCD) and integrate them with streams data from post-process feature.

Isolation Forest(IForest): The main advantage of Isolation Forest is that this algorithm is proven effective in detecting anomaly detection. ROC and Precision table show that IForest is stable. The minor problem is that the IForest training time is slightly slower than HBOS and MCD. However, this problem can be solved with the multithreading method.

HBOS: This algorithm is faster than IForest, but the ROC results show that this algorithm struggles to detect anomalies in different sensors type.

MCD: similar to HBOS, this algorithm is faster than IForest. However, the precision score of this algorithm is low.

We use **Isolation Forest**, because its reliability and stability.

### Testing

#### How to run

### Result
These are the results of two different tuning on anomaly detection program using temperature sensor.
#### High sensitivity
    Parameter:
    1. Outlier_fraction: 0.03
    2. Anomaly_score_threshold: 2.45
    3. Upper_threshold: 40
    4. Lower_threshold: 0
    5. Train_number: 5760 (*4 days)
    6. Batch_size: 60 (*1 hours)
    Note: *in minute
![sliding_window](/docs/sliding_window.png)
#### Low sensitivity
    Parameter:
    1. Outlier_fraction: 0.03
    2. Anomaly_score_threshold: 3
    3. Upper_threshold: 40
    4. Lower_threshold: 0 
    5. Train_number: 20160 (*14 days)
    6. Batch_size: 60 (*1 hours)
    Note: *in minute
![sliding_window](/docs/sliding_window.png)
