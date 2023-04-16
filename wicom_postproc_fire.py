from pymodbus.client.sync import ModbusSerialClient as ModbusRTU
import mysql.connector as SQL
import datetime
import time
import numpy as np
from sklearn.ensemble import IsolationForest

db_host = '113.198.211.95'
db_name = 'PSU650'
table_name = 'Coex'
anomaly_table_name = 'Coex_Anomaly'
slave = ModbusRTU(method='rtu', port='/dev/ttyUSB0', bytesize=8, stopbits=1, parity='N', baudrate=115200)
connection = slave.connect()

mydb = SQL.connect(host=db_host, user='admina', password='admina', database=db_name)
mycursor = mydb.cursor()

# Initialize the sliding window and adaptive threshold
window_size = 3600 * 6  # 6 hours
window_start_time = datetime.datetime.now()
window_data = {'Temp': [], 'Hum': [], 'PM0.3': [], 'PM0.5': [], 'PM1.0': [], 'PM2.5': [], 'PM5.0': [], 'PM10.0': []}
thresholds = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
anomaly_scores = np.zeros(8)
n_samples = 0

while True:
    start_time = datetime.datetime.now()

    try:
        # Read the data from the Modbus slave
        result = slave.read_input_registers(address=10, count=14, unit=1)
        time_stamp = datetime.datetime.now()
        Temp = float(result.registers[0]/100)
        Hum = float(result.registers[1]/100)
        PM03, PM05, PM10, PM25, PM50, PM100 = [float(result.registers[i]) for i in range(3, 14, 2)]

        # Calculate the anomaly score for each variable using the Isolation Forest
        if n_samples > 0:
            anomaly_scores = np.zeros(8)
            for i, var in enumerate(window_data.keys()):
                X = np.reshape(window_data[var], (-1, 1))
                clf = IsolationForest(random_state=0).fit(X)
                anomaly_scores[i] = clf.score_samples(np.reshape(eval(var), (1, -1)))[0]

        # Insert the data into the Coex table
        query = f'INSERT INTO `{table_name}`(`Time`, `Temp`, `Hum`, `PM0.3`, `PM0.5`, `PM1.0`, `PM2.5`, `PM5.0`, `PM10.0`) \
                  VALUES ("{time_stamp}","{Temp}","{Hum}","{PM03}","{PM05}","{PM10}","{PM25}","{PM50}","{PM100}")'
        mycursor.execute(query)
        mydb.commit()

        # Insert the anomaly scores into the Coex_Anomaly table
        query = f'INSERT INTO `{anomaly_table_name}`(`Time`, `Temp_Anomaly`, `Hum_Anomaly`, `PM0.3_Anomaly`, `PM0.5_Anomaly`, `PM1.0_Anomaly`, `PM2.5_Anomaly`, `PM5.0_Anomaly`, `PM10.0_Anomaly`) \
                  VALUES ("{time_stamp}", "{anomaly_scores[0]}", "{anomaly_scores[1]}", "{anomaly_scores[2]}", "{anomaly_scores[3]}", "{anomaly_scores[4]}", "{anomaly_scores[5]}", "{anomaly_scores[6]}", "{anomaly_scores[7]}")'
        mycursor.execute(query)
        mydb.commit()
        # Add the new data to the sliding window
        window_data['Temp'].append(Temp)
        window_data['Hum'].append(Hum)
        window_data['PM0.3'].append(PM03)
        window_data['PM0.5'].append(PM05)
        window_data['PM1.0'].append(PM10)
        window_data['PM2.5'].append(PM25)
        window_data['PM5.0'].append(PM50)
        window_data['PM10.0'].append(PM100)

        # Check if the sliding window needs to be shifted
        if (time_stamp - window_start_time).total_seconds() >= window_size:
            # Remove the oldest data from the sliding window
            window_data['Temp'].pop(0)
            window_data['Hum'].pop(0)
            window_data['PM0.3'].pop(0)
            window_data['PM0.5'].pop(0)
            window_data['PM1.0'].pop(0)
            window_data['PM2.5'].pop(0)
            window_data['PM5.0'].pop(0)
            window_data['PM10.0'].pop(0)

            # Update the start time of the sliding window
            window_start_time = time_stamp

        # Update the number of samples
        n_samples += 1

    except Exception as e:
        print(f'Error: {e}')

    # Sleep for the remaining time to maintain a consistent sampling frequency
    end_time = datetime.datetime.now()
    sleep_time = max(0, 1 - (end_time - start_time).total_seconds())
    time.sleep(sleep_time)
