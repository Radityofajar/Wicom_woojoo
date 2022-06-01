# Wicom_woojoo

alldata_new.csv >> Raw data 
([col_date,temperature1,temperature2,temperature3,humidity1,humidity2,waterlevel,leakage,fire,isclosed])

Woojoo_data_X.csv >> Isolation forest result for each variable
([col_date,humidity1,anomaly_hum1])

for **anomaly = 1: is normal and -1: is abnormal**

Woojoo_alldata.csv >> Data & Isolation forest result
([col_date,temperature1,temperature2,temperature3,humidity1,humidity2,waterlevel,leakage,fire,isclosed,anomaly_temp1,anomaly_temp2,anomaly_temp3,anomaly_hum1,anomaly_hum2,anomaly_waterlevel,anomaly_waterleak,anomaly_fire,anomaly_door]
