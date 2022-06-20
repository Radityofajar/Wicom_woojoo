from pyiotown import post
from pyiotown_wicom import postprocess
import json
import sys
import warnings
warnings.filterwarnings('ignore')

def receive_data(rawdata):
    raw_data = rawdata['data']
    #take the data
    print(raw_data)
    test = len(raw_data.split())
    if raw_data == 'Missing':
        details = 'Data is not complete'
        print(details)
    elif test >= 11:
        #take only useful information
        msg = raw_data.split()
        datasensor = str(msg[11:])
        datasensor = datasensor.replace(']', '')
        datasensor = datasensor.replace('[', '')
        datasensor = datasensor.replace("', '", '')
        datasensor = datasensor.replace("'", '')
        pythonObj = json.loads(json.loads(json.dumps(datasensor)))
        #change to json format
        sensor = pythonObj['data']
    return sensor

def post_process(rawdata):
    global anomaly_threshVal0

    #receive data from sensor
    message = receive_data(rawdata=rawdata)
    #print(message)
    sensor_type = message['dtype']
    if sensor_type == 'wlak':
        sensor_wlak = message['val0']

        if float(sensor_wlak) == 0:
            #normal condition
            sensor_wlak_status = 'normal/no_wlak'
        else:
            #abnormal condition
            sensor_wlak_status = 'abnormal/wlak'

        #store the data in order to send it back to IoT.own
        changedata = {}
        changedata['dtype'] = message['dtype']
        changedata['nid'] = message['nid']
        changedata['result_wlak'] = sensor_wlak_status
        rawdata['data'] = changedata
        print(rawdata)
        return rawdata
    else:
        print('Sensor is not supported')

if __name__ == '__main__':
    if len(sys.argv) != 11:
        print(f"Usage: {sys.argv[0]} [URL] [name] [token] [outlier_fraction] [anomaly_threshVal0] [sensor ID]")
        exit(1)

    #IoT.own setting
    postproc_name = 'wlak'
    url = sys.argv[1]
    username = sys.argv[2]
    password = sys.argv[3]

    #model parameter
    outlier_fraction = float(sys.argv[4])

    #clustering setting
    anomaly_threshVal0 = float(sys.argv[5])
    
    postprocess(url,postproc_name,post_process, username, password)