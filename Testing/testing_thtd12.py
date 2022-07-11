from pyiotown import post
import pandas as pd
import time
import numpy as np
from joblib import load, dump
from collections import Counter
import threading
from sklearn.ensemble import IsolationForest
import csv
import warnings
warnings.filterwarnings('ignore')

data_temp1 = pd.read_csv('temperature1.csv')
data_temp1 = data_temp1['temperature1']

data_hum1 = pd.read_csv('humidity1.csv')
data_hum1 = data_hum1['humidity1']

nid_library = {}
nid_library['data'] = np.array([[]]) #make a new array for new nid
nid_library['anomaly_score'] = np.array([[]])
nid_library['anomaly_status'] = np.array([[]])
arr_data_temp1 = np.array([[]])
arr_score_temp1 = np.array([[]])
arr_status_temp1 = np.array([[]])
arr_threshold_temp1 = np.array([[]])

nid_library_2 = {}
nid_library_2['data'] = np.array([[]]) #make a new array for new nid
nid_library_2['anomaly_score'] = np.array([[]])
nid_library_2['anomaly_status'] = np.array([[]])
arr_data_hum1 = np.array([[]])
arr_score_hum1 = np.array([[]])
arr_status_hum1 = np.array([[]])
arr_threshold_hum1 = np.array([[]])
counter = 1

batch_size = 30
train_number = 14*1440
outlier_fraction_param = 0.03
anomaly_threshVal0_param = 3.5
anomaly_threshVal1_param = 3.5
threshold_temp1_lower = 0
threshold_temp1_higher = 40
threshold_hum1_lower = 0
threshold_hum1_higher = 60

def train(outlier_fraction1, outlier_fraction2):
    t1 = time.time()
    #model setting
    estimator = 100
    samples = 1000
    randstate = 42

    #outlier parameter
    if outlier_fraction1 == 0:
        outlier_fraction1 = 0.01 # 1% of contamination
    elif outlier_fraction1 >= outlier_fraction_param:
        outlier_fraction1 = outlier_fraction_param
    else:
        outlier_fraction1 = outlier_fraction1

    if outlier_fraction2 == 0:
        outlier_fraction2 = 0.01 # 1% of contamination
    elif outlier_fraction2 >= outlier_fraction_param:
        outlier_fraction2 = outlier_fraction_param
    else:
        outlier_fraction2 = outlier_fraction2

    #model initialization
    model_temp1 = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction1)
    model_hum1 = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction2)

    #data preprocess
    nid_library['data'] = nid_library['data'].reshape(-1,1)
    nid_library_2['data'] = nid_library_2['data'].reshape(-1,1)

    #model training
    model_temp1.fit(nid_library['data'])
    model_hum1.fit(nid_library_2['data'])

    #filename
    var1 = 'model\model_'
    var_hum = '_hum1.joblib'
    var_temp = '_temp1.joblib'
    filename_hum_model = var1 + 'testing12' + var_hum
    filename_temp_model = var1 + 'testing12' + var_temp

    #save/overwrite model
    dump(model_hum1, filename_hum_model)
    dump(model_temp1, filename_temp_model)
    #print(time.time()-t1)
    #print('update the model')
i = 0
while i<len(data_temp1):
    try:
        #print(i)
        t2 = time.time()
        data_temperature = np.array([data_temp1[i]]).T
        data_humidity = np.array([data_hum1[i]]).T
        #print(data)
        nid_library['data'] = np.append(nid_library['data'], data_temperature)
        arr_data_temp1 = np.append(arr_data_temp1, data_temperature)

        nid_library_2['data'] = np.append(nid_library_2['data'], data_humidity)
        arr_data_hum1 = np.append(arr_data_hum1, data_humidity)

        if counter == 1:
                #mode1: using initial mode
                try: #if spesified model is already built
                    #filename
                    var1 = 'model\model_'
                    var_hum = '_hum1.joblib'
                    var_temp = '_temp1.joblib'
                    filename_hum_model = var1 + 'testing12' + var_hum
                    filename_temp_model = var1 + 'testing12' + var_temp
                    #load model
                    model_temp1 = load(filename_temp_model) 
                    model_hum1 = load(filename_hum_model)

                except: #if there is no spesificied model
                    #filename
                    filename_temp_model = 'model\model_temp1.joblib'
                    filename_hum_model = 'model\model_hum1.joblib'
                    #load model
                    model_temp1 = load(filename_temp_model)
                    model_hum1 = load(filename_hum_model)
                    print('Take initial model')
                else:
                    print('Using specified model')
                finally:
                    print(filename_temp_model)
                    print(filename_hum_model)
                    anomaly_threshVal0 = 0.0
                    anomaly_threshVal1 = 0.0
                    counter += 1

        elif counter <= batch_size :
            #mode2: Keep using initial model until the data stored in array
            counter += 1

        elif counter == (batch_size + 1):
            #mode 3: retrain the model

            #calculate the outlier fraction
            outlier1 = Counter(nid_library['anomaly_score']) #temp
            outlier2 = Counter(nid_library_2['anomaly_score']) #hum
            outlier_fraction1 = outlier1['abnormal'] / len(nid_library['anomaly_score']) #temp
            outlier_fraction2 = outlier2['abnormal'] / len(nid_library_2['anomaly_score']) #hum

            #multithreading
            thread = threading.Thread(target=train, args=(outlier_fraction1, outlier_fraction2))
            if thread.is_alive():
                print('thread still running')
            else:
                #print('thread is starting')
                thread.start()
            counter += 1
            thread.join()
        
        elif counter == (batch_size+2):
            #model 4: load retrain model
            #filename
                var1 = 'model\model_'
                var_hum = '_hum1.joblib'
                var_temp = '_temp1.joblib'
                filename_hum_model = var1 + 'testing12' + var_hum
                filename_temp_model = var1 + 'testing12' + var_temp

                #load model
                mmodel_temp1 = load(filename_temp_model)
                model_hum1 = load(filename_hum_model)
                #print('model loaded')

                #calculate the anomaly score threshold
                anomaly_score_temp_mean = nid_library['anomaly_score'].mean()
                anomaly_score_temp_std = nid_library['anomaly_score'].std()
                anomaly_score_temp_cal = anomaly_score_temp_mean - (anomaly_score_temp_std*anomaly_threshVal0_param)
                
                if anomaly_score_temp_cal <= -0.15:
                    anomaly_threshVal0 = -0.15
                elif anomaly_score_temp_cal >= 0.01:
                    anomaly_threshVal0 = 0.01
                else:
                    anomaly_threshVal0 = anomaly_score_temp_cal

                #calculate the anomaly score threshold for humidity
                anomaly_score_hum_mean = nid_library_2['anomaly_score'].mean()
                anomaly_score_hum_std = nid_library_2['anomaly_score'].std()
                anomaly_score_hum_cal = anomaly_score_hum_mean - (anomaly_score_hum_std*anomaly_threshVal1_param)
                
                if anomaly_score_hum_cal <= -0.15:
                    anomaly_threshVal1 = -0.15
                if anomaly_score_hum_cal >= 0.01:
                    anomaly_threshVal1 = 0.01
                else:
                    anomaly_threshVal1 = anomaly_score_hum_cal
                
                counter += 1

        elif counter <= (batch_size + batch_size):
            #mode 5: sliding window method
            counter += 1

        else:
            #optimize the array size of sliding window
            nid_library['data'] = nid_library['data'][-(train_number+2*batch_size):]
            nid_library['anomaly_score'] = nid_library['anomaly_score'][-(train_number+2*batch_size):]
            nid_library['anomaly_status'] = nid_library['anomaly_status'][-(train_number+2*batch_size):]
            #optimize the array size of sliding window for humidity
            nid_library_2['data'] = nid_library_2['data'][-(train_number+2*batch_size):]
            nid_library_2['anomaly_score'] = nid_library_2['anomaly_score'][-(train_number+2*batch_size):]
            nid_library_2['anomaly_status'] = nid_library_2['anomaly_status'][-(train_number+2*batch_size):]
            counter = (batch_size+1)
            #print('optimize array')

        #preprocess the data for anomaly detection
        data_temperature_reshape = data_temperature.reshape(1,-1)
        data_humidity_reshape = data_humidity.reshape(1,-1)

        #anomaly detection / Isoloation forest prediction
        anomaly_score_temp = model_temp1.decision_function(data_temperature_reshape)
        anomaly_score_hum = model_hum1.decision_function(data_humidity_reshape)

        #print(anomaly_score_wlvl)
        #print(anomaly_threshVal0)
        #print(sensor_wlvl[0])

        #clustering between normal & abnormal
        #Water level sensor
        if anomaly_score_temp > anomaly_threshVal0 and float(data_temperature[0]) > threshold_temp1_lower and float(data_temperature[0]) < threshold_temp1_higher:
            #normal condition
            sensor_temp_status = 'normal'
        else:
            #abnormal condition
            sensor_temp_status = 'abnormal'
        #humidity sensor
        if anomaly_score_hum > anomaly_threshVal1 and float(data_humidity[0]) > threshold_hum1_lower and float(data_humidity[0]) < threshold_hum1_higher:
            #normal condition
            sensor_hum_status = 'normal'
        else:
            #abnormal condition
            sensor_hum_status = 'abnormal'

        nid_library['anomaly_score'] = np.append(nid_library['anomaly_score'],float(anomaly_score_temp))
        nid_library['anomaly_status'] = np.append(nid_library['anomaly_status'],sensor_temp_status)

        nid_library_2['anomaly_score'] = np.append(nid_library_2['anomaly_score'],float(anomaly_score_hum))
        nid_library_2['anomaly_status'] = np.append(nid_library_2['anomaly_status'],sensor_hum_status)

        arr_score_temp1 = np.append(arr_score_temp1,anomaly_score_temp)
        arr_status_temp1 = np.append(arr_status_temp1,sensor_temp_status)
        arr_threshold_temp1 = np.append(arr_threshold_temp1, anomaly_threshVal0)

        arr_score_hum1 = np.append(arr_score_hum1,anomaly_score_hum)
        arr_status_hum1 = np.append(arr_status_hum1,sensor_hum_status)
        arr_threshold_hum1 = np.append(arr_threshold_hum1, anomaly_threshVal1)

        print(data_temp1[i])
        print('anomaly_score_temp:' + str(anomaly_score_temp))
        print('anomaly_threshold_temp:' + str(anomaly_threshVal0))
        print('anomaly_status_temp:' + sensor_temp_status)
        print(data_hum1[i])
        print('anomaly_score_hum:' + str(anomaly_score_hum))
        print('anomaly_threshold_hum:' + str(anomaly_threshVal1))
        print('anomaly_status_hum:' + sensor_hum_status)

        #print(time.time()-t2)
        time.sleep(0.5)
    except:
        np.savetxt("thtd12/result_data_temp1.csv", arr_data_temp1, delimiter=",", fmt="%.1f")
        np.savetxt("thtd12/result_anomaly_score_temp1.csv", arr_score_temp1, delimiter=",", fmt="%.3f")
        np.savetxt("thtd12/result_anomaly_threshold_temp1.csv", arr_threshold_temp1, delimiter=",", fmt="%.3f")
        arr_status_temp1.tofile('thtd12/result_anomaly_status_temp1.csv', sep='\n')

        np.savetxt("thtd12/result_data_hum1.csv", arr_data_hum1, delimiter=",", fmt="%.1f")
        np.savetxt("thtd12/result_anomaly_score_hum1.csv", arr_score_hum1, delimiter=",", fmt="%.3f")
        np.savetxt("thtd12/result_anomaly_threshold_hum1.csv", arr_threshold_hum1, delimiter=",", fmt="%.3f")
        arr_status_hum1.tofile('thtd12/result_anomaly_status_hum1.csv', sep='\n')
        time.sleep(1)
        print('except')
    finally:
        i += 1

np.savetxt("thtd12/result_data_temp1.csv", arr_data_temp1, delimiter=",", fmt="%.1f")
np.savetxt("thtd12/result_anomaly_score_temp1.csv", arr_score_temp1, delimiter=",", fmt="%.3f")
np.savetxt("thtd12/result_anomaly_threshold_temp1.csv", arr_threshold_temp1, delimiter=",", fmt="%.3f")
arr_status_temp1.tofile('thtd12/result_anomaly_status_temp1.csv', sep='\n')

np.savetxt("thtd12/result_data_hum1.csv", arr_data_hum1, delimiter=",", fmt="%.1f")
np.savetxt("thtd12/result_anomaly_score_hum1.csv", arr_score_hum1, delimiter=",", fmt="%.3f")
np.savetxt("thtd12/result_anomaly_threshold_hum1.csv", arr_threshold_hum1, delimiter=",", fmt="%.3f")
arr_status_hum1.tofile('thtd12/result_anomaly_status_hum1.csv', sep='\n')
print('finish')