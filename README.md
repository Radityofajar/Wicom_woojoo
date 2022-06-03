# Wicom_woojoo

### System Requirement
    Python 3
    Pyiotown >> python3 -m pip install pyiotown
    sklearn >>  pip install sklearn
    PyOD >> pip install pyod    
    joblib >> pip install joblib
    numpy >> pip install numpy
    pandas >> pip install pandas 
### Data
Contains all the datasets that collected from sensor located in WicomAI LAB, Kookmin University
### model
Contains initial models (temperature, humidity, waterlevel, waterleak, door, fire)

We have 3 different models options for anomaly detection: **Isolation Forest (IForest)**, **Histogram- based outlier detection (HBOS)**, and **Minimum Covariance Determinant (MCD)**.
### postproc_IF.py
This is the main program to run the anomaly detection based on Isolation Forest (IForest) using postprocess features by IoT.own Coxlab.
(nb: flow_postproc.py is similar with postproc.py but more readable and easily modify)

  Step 1: Prepare the initial model using model_init.py
  
  Step 2: open postproc.py and change parameter as needed
  
    -> Thresholding value parameter: to create upper and lower boundaries.
    
    -> sliding window parameter: to keep the uniqueness and rarity.
    
    -> model initialization parameter: to use what kind of model needed and also how you want to retrain it.
  Step 3: Run the postproc.py program
  
    -> Parameters
    
      1. Status: The systems condition. the value whether normal(1) or abnormal(-1)
      
      2. Anomaly score: How likely the value to be normal or abnormal. When it is closer to -1, then it is likely to be abnormal and vice versa.

#### Sliding window
Parameter: batch_size and train_number
![sliding_window](/docs/sliding_window.png)

### postproc_HBOS.py
Histogram- based outlier detection (HBOS) is an efficient unsupervised method. It assumes the feature independence and calculates the degree of outlyingness by building histograms. (model_init_2.py)

  -> Parameter
  
    1. Status: The systems condition. the value whether normal(0) or abnormal(1)
    
    2. Anomaly score: How likely the value to be normal or abnormal. Outliers are assigned with larger anomaly scores.

### postproc_MCD.py
Outlier Detection with Minimum Covariance Determinant (MCD) estimator is to be applied on Gaussian-distributed data, but could still be relevant on data drawn from a unimodal, symmetric distribution.
(model_init_2.py)

  -> Parameter
  
    1. Status: The systems condition. the value whether normal(0) or abnormal(1)
    
    2. Anomaly score: How likely the value to be normal or abnormal. Outliers are assigned with larger anomaly scores.
    
## Comparison & Conclusion
![model_comparison1](/docs/result_timecomplexity.JPG)
![model_comparison2](/docs/result_roc.JPG)
![model_comparison3](/docs/result_precision.JPG)

The tables above show the comparison between each anomaly detection algorithm. We choose the best 3 among of them (Isolation Forest, HBOS, and MCD) and integrate it with streams data from postprocess feature.

Isolation Forest(IForest): The main advantage of Isolation Forest is this algorithm proven effective to detect anomaly detection. ROC and Precision table show that IForest is pretty stable. The minor problem is IForest training time little bit slower compare with HBOS and MCD. However, this problem can be solved with multithreading method.

HBOS: This algorithm is faster than IForest, but the ROC results show that this algorithm little bit struggle to detect anomaly in different types of data.

MCD: similar like HBOS, this algorithm is faster than IForest. However, the precision score of this algorithm is low.

We strongly recommend to use **Isolation Forest**, because its reliability and stability.

