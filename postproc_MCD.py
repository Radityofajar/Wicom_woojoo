#import all the library needed
import numpy as np
from pyiotown_wicom import postprocess
import pyiotown.post
from joblib import load, dump
from pyod.models.mcd import MCD
from sklearn.preprocessing import StandardScaler
import threading
import warnings
warnings.filterwarnings("ignore")

counter = 0

#thresholding value
threshold_waterlevel1 = 300
threshold_waterlevel2 = 3500
threshold_hum1_lower = 15
threshold_hum1_upper = 55
threshold_hum2_lower = 15
threshold_hum2_upper = 55
threshold_temp1_lower = 5
threshold_temp1_upper = 40
threshold_temp2_lower = 5
threshold_temp2_upper = 40
threshold_temp3_lower = 5
threshold_temp3_upper = 40

#sliding window setting
batch_size = 60 # 60 = 1 hour
train_number = 1440 # 1440 = 1 day


def train(): # for retraining model & overwriting model
    global arr_hum1, arr_hum2, arr_hum1_norm, arr_hum2_norm
    global arr_temp1, arr_temp2, arr_temp3, arr_temp1_norm, arr_temp2_norm, arr_temp3_norm
    global arr_waterlevel, arr_waterleak, arr_waterlevel_norm
    global arr_door, arr_fire
    outliers_fraction = 0.08

    #model initialization
    model_hum1 = MCD(contamination=outliers_fraction,random_state=42)
    model_hum2 = MCD(contamination=outliers_fraction,random_state=42)
    model_temp1 = MCD(contamination=outliers_fraction,random_state=42)
    model_temp2 = MCD(contamination=outliers_fraction,random_state=42)
    model_temp3 = MCD(contamination=outliers_fraction,random_state=42)
    model_waterlevel = MCD(contamination=outliers_fraction,random_state=42)

    #data preprocess
    arr_hum1 = arr_hum1.reshape(-1,1)
    arr_hum2 = arr_hum2.reshape(-1,1)
    arr_temp1 = arr_temp1.reshape(-1,1)
    arr_temp2 = arr_temp2.reshape(-1,1)
    arr_temp3 = arr_temp3.reshape(-1,1)
    arr_waterlevel = arr_waterlevel.reshape(-1,1)

    arr_hum1_norm = arr_hum1_norm.reshape(-1,1)
    arr_hum2_norm = arr_hum2_norm.reshape(-1,1)
    arr_temp1_norm = arr_temp1_norm.reshape(-1,1)
    arr_temp2_norm = arr_temp2_norm.reshape(-1,1)
    arr_temp3_norm = arr_temp3_norm.reshape(-1,1)
    arr_waterlevel_norm = arr_waterlevel_norm.reshape(-1,1)
 
    sc_hum1 = StandardScaler().fit_transform(arr_hum1)
    sc_hum2 = StandardScaler().fit_transform(arr_hum2)
    sc_temp1 = StandardScaler().fit_transform(arr_temp1)
    sc_temp2 = StandardScaler().fit_transform(arr_temp2)
    sc_temp3 = StandardScaler().fit_transform(arr_temp3)
    sc_waterlevel = StandardScaler().fit_transform(arr_waterlevel)

    #model training
    model_hum1.fit(arr_hum1_norm)
    model_hum2.fit(arr_hum2_norm)
    model_temp1.fit(arr_temp1_norm)
    model_temp2.fit(arr_temp2_norm)
    model_temp3.fit(arr_temp3_norm)
    model_waterlevel.fit(arr_waterlevel_norm)

    #save/overwrite model
    dump(model_hum1, 'model\MCD_model_hum1.joblib')
    dump(model_hum2, 'model\MCD_model_hum2.joblib')
    dump(model_temp1, 'model\MCD_model_temp1.joblib')
    dump(model_temp2, 'model\MCD_model_temp2.joblib')
    dump(model_temp3, 'model\MCD_model_temp3.joblib')
    dump(model_waterlevel, 'model\MCD_model_waterlevel.joblib')

    dump(sc_hum1, 'SC\std_scaler_hum1.bin')
    dump(sc_hum2, 'SC\std_scaler_hum2.bin')
    dump(sc_temp1, 'SC\std_scaler_temp1.bin')
    dump(sc_temp2, 'SC\std_scaler_temp2.bin')
    dump(sc_temp3, 'SC\std_scaler_temp3.bin')
    dump(sc_waterlevel, 'SC\std_scaler_waterlevel.bin')

    print('Retraining is done')

def post_process(message):

    global arr_hum1, arr_hum2, arr_hum1_norm, arr_hum2_norm
    global arr_temp1, arr_temp2, arr_temp3, arr_temp1_norm, arr_temp2_norm, arr_temp3_norm
    global arr_waterlevel, arr_waterleak, arr_waterlevel_norm, arr_waterleak_norm
    global arr_door, arr_fire, arr_door_norm, arr_fire_norm
    global counter
    global model_hum1, model_hum2
    global model_temp1, model_temp2, model_temp3
    global model_waterlevel

    print(message)
    
    temp1 = np.array([message['data']['temp1']]).T
    temp2 = np.array([message['data']['temp2']]).T
    temp3 = np.array([message['data']['temp3']]).T
    hum1 = np.array([message['data']['hum1']]).T
    hum2 = np.array([message['data']['hum2']]).T
    water_level = np.array([message['data']['water_level']]).T
    water_leak = np.array([message['data']['water_leak']]).T
    door = np.array([message['data']['door']]).T
    fire = np.array([message['data']['fire']]).T
    
    print(counter)

    if counter == 0: #Using Initial model for the mode 1 & 2
        print("mode1")
        model_hum1 = load("model\MCD_model_hum1.joblib")
        model_hum2 = load("model\MCD_model_hum2.joblib")
        model_temp1 = load("model\MCD_model_temp1.joblib")
        model_temp2 = load("model\MCD_model_temp2.joblib")
        model_temp3 = load("model\MCD_model_temp3.joblib")
        model_waterlevel = load("model\MCD_model_waterlevel.joblib")

        sc_hum1 = load('SC\std_scaler_hum1.bin')
        sc_hum2 = load('SC\std_scaler_hum2.bin')
        sc_temp1 = load('SC\std_scaler_temp1.bin')
        sc_temp2 = load('SC\std_scaler_temp2.bin')
        sc_temp3 = load('SC\std_scaler_temp3.bin')
        sc_waterlevel = load('SC\std_scaler_waterlevel.bin')

        counter += 1 

    elif counter<=train_number: 
        #keep using initial model until the data stored in array is enough
        print("mode2")
        counter +=1   

    elif counter == (train_number+1) : 
        #retrain the  model
        thread = threading.Thread(target=train)
        print("mode3")
        print(thread.is_alive())
        if thread.is_alive():
            print('thread still running')          
        else:
            print('thread is starting')
            thread.start()
        counter += 1
        thread.join()

    elif counter<= (train_number + batch_size): 
        #sliding window method
        print("mode4")
        counter +=1
    
    else: #optimize the array size of the sliding window
        counter = (train_number+1)
        arr_hum1 = arr_hum1[-train_number:]
        arr_hum2 = arr_hum2[-train_number:]
        arr_temp1 = arr_temp1[-train_number:]
        arr_temp3 = arr_temp3[-train_number:]
        arr_waterlevel = arr_waterlevel[-train_number:]
        arr_waterleak = arr_waterleak[-train_number:]
        arr_fire = arr_fire[-train_number:]
        arr_door = arr_door[-train_number:]

        arr_hum1_norm = arr_hum1_norm[-train_number:]
        arr_hum2_norm = arr_hum2_norm[-train_number:]
        arr_temp1_norm = arr_temp1_norm[-train_number:]
        arr_temp3_norm = arr_temp3_norm[-train_number:]
        arr_waterlevel_norm = arr_waterlevel_norm[-train_number:]

    #normalize the data using standard scaler
    hum1_norm = sc_hum1.transform(hum1.reshape(1,-1))
    hum2_norm = sc_hum2.transform(hum2.reshape(1,-1))
    temp1_norm = sc_temp1.transform(temp1.reshape(1,-1))
    temp2_norm = sc_temp2.transform(temp2.reshape(1,-1))
    temp3_norm = sc_temp3.transform(temp3.reshape(1,-1))
    water_level_norm = sc_waterlevel.transform(water_level.reshape(1,-1))


    #input data to the window
    arr_hum1 = np.append(arr_hum1,hum1)
    arr_hum2 = np.append(arr_hum2,hum2)
    arr_temp1 = np.append(arr_temp1,temp1)
    arr_temp2 = np.append(arr_temp2,temp2)
    arr_temp3 = np.append(arr_temp3,temp3)
    arr_waterlevel = np.append(arr_waterlevel,water_level)
    arr_waterleak = np.append(arr_waterleak,water_leak)
    arr_fire = np.append(arr_fire,fire)
    arr_door = np.append(arr_door,door)

    arr_hum1_norm = np.append(arr_hum1_norm,hum1_norm)
    arr_hum2_norm = np.append(arr_hum2_norm,hum2_norm)
    arr_temp1_norm = np.append(arr_temp1_norm,temp1_norm)
    arr_temp2_norm = np.append(arr_temp2_norm,temp2_norm)
    arr_temp3_norm = np.append(arr_temp3_norm,temp3_norm)
    arr_waterlevel_norm = np.append(arr_waterlevel_norm,water_level_norm)


    #preprocess the data for anomaly detection
    newhum1 = hum1_norm.reshape(1,-1)
    newhum2 = hum2_norm.reshape(1,-1)
    newtemp1 = temp1_norm.reshape(1,-1)
    newtemp2 = temp2_norm.reshape(1,-1)
    newtemp3 = temp3_norm.reshape(1,-1)
    newwater_level = water_level_norm.reshape(1,-1)

    
    #anomaly detection / Isolation Forest Prediction
    anomaly_score_hum1 = model_hum1.decision_function(newhum1)
    anomaly_score_hum2 = model_hum2.decision_function(newhum2)
    anomaly_score_temp1 = model_temp1.decision_function(newtemp1)
    anomaly_score_temp2 = model_temp2.decision_function(newtemp2)
    anomaly_score_temp3 = model_temp3.decision_function(newtemp3)
    anomaly_score_waterlevel = model_waterlevel.decision_function(newwater_level)


    anomalies_hum1 = model_hum1.predict(newhum1)
    anomalies_hum2 = model_hum2.predict(newhum2)
    anomalies_temp1 = model_temp1.predict(newtemp1)
    anomalies_temp2 = model_temp2.predict(newtemp2)
    anomalies_temp3 = model_temp3.predict(newtemp3)
    anomalies_waterlevel = model_waterlevel.predict(newwater_level)


    #clustering between normal & abnormal
    if anomalies_hum1 == 0 and float(hum1[0]) > threshold_hum1_lower and float(hum1[0]) < threshold_hum1_upper:
        status_hum1 = 'normal'
    else:
        status_hum1 = 'abnormal'
    
    if anomalies_hum2 == 0 and float(hum2[0]) > threshold_hum2_lower and float(hum2[0]) < threshold_hum2_upper:
        status_hum2 = 'normal'
    else:
        status_hum2 = 'abnormal'

    if anomalies_temp1 == 0 and float(temp1[0]) > threshold_temp1_lower and float(temp1[0]) < threshold_temp1_upper:
        status_temp1 = 'normal'
    else:
        status_temp1 = 'abnormal'

    if anomalies_temp2 == 0 and float(temp2[0]) > threshold_temp2_lower and float(temp2[0]) < threshold_temp2_upper:
        status_temp2 = 'normal'
    else:
        status_temp2 = 'abnormal'

    if anomalies_temp3 == 0 and float(temp3[0]) > threshold_temp3_lower and float(temp3[0]) < threshold_temp3_upper:
        status_temp3 = 'normal'
    else:
        status_temp3 = 'abnormal'

    if anomalies_waterlevel == 0 and float(water_level[0]) > threshold_waterlevel1 and float(water_level[0]) < threshold_waterlevel2:
        status_waterlevel = 'normal'
    else:
        status_waterlevel = 'abnormal'

    if float(water_leak[0]) == 0:#thresholding for binary sensor
        status_waterleak = 'normal'
    else:
        status_waterleak = 'abnormal/isflood'
    
    if float(fire[0]) == 0: #thresholding for binary sensor
        
        status_fire = 'normal'
    else:
        status_fire = 'abnormal/fire'

    if float(door[0]) == 0: #thresholding for binary sensor
        status_door = 'normal'
    else:
        status_door = 'abnormal/open'

    changedata = {}

    #store the data in order to send it back to IoT.own
    changedata['status_temp1'] = status_temp1
    changedata['status_temp2'] = status_temp2
    changedata['status_temp3'] = status_temp3
    changedata['status_hum1'] = status_hum1
    changedata['status_hum2'] = status_hum2
    changedata['status_water_level'] = status_waterlevel
    changedata['status_water_leak'] = status_waterleak
    changedata['status_door'] = status_door
    changedata['status_fire'] = status_fire

    changedata['temp1'] = float(temp1[0])
    changedata['temp2'] = float(temp2[0])
    changedata['temp3'] = float(temp3[0])
    changedata['hum1'] = float(hum1[0])
    changedata['hum2'] = float(hum2[0])
    changedata['water_level'] = float(water_level[0])
    changedata['water_leak'] = float(water_leak[0])
    changedata['door'] = float(door[0])
    changedata['fire'] = float(fire[0])
    
    changedata['anomaly_score_temp1'] = round(float(anomaly_score_temp1[0]),2)
    changedata['anomaly_score_temp2'] = round(float(anomaly_score_temp2[0]),2)
    changedata['anomaly_score_temp3'] = round(float(anomaly_score_temp3[0]),2)
    changedata['anomaly_score_hum1'] = round(float(anomaly_score_hum1[0]),2)
    changedata['anomaly_score_hum2'] = round(float(anomaly_score_hum2[0]),2)
    changedata['anomaly_score_waterlevel'] = round(float(anomaly_score_waterlevel[0]),2)
 
    message['data'] = changedata
    print(changedata)
    return message

if __name__ == '__main__':
    arr_hum1 = np.array([[]])
    arr_hum2 = np.array([[]])
    arr_temp1 = np.array([[]])
    arr_temp2 = np.array([[]])
    arr_temp3 = np.array([[]])
    arr_waterlevel = np.array([[]])
    arr_waterleak = np.array([[]])
    arr_door = np.array([[]])
    arr_fire = np.array([[]])

    arr_hum1_norm = np.array([[]])
    arr_hum2_norm = np.array([[]])
    arr_temp1_norm = np.array([[]])
    arr_temp2_norm = np.array([[]])
    arr_temp3_norm = np.array([[]])
    arr_waterlevel_norm = np.array([[]])
    arr_waterleak_norm = np.array([[]])
    arr_door_norm = np.array([[]])
    arr_fire_norm = np.array([[]])

    postproc_name = 'PostProcessExample3'
    url = "https://town.coxlab.kr/"
    username = "rfpamungkas23@gmail.com"
    password = "c34859e08fa526f642881820d5108ccd475d5b58efbc8b4a5b89fd93366fe1d1"
    #postprocess(url,postproc_name,post_process, username, password)
    pyiotown.post.postprocess(url,postproc_name,post_process, username, password)
