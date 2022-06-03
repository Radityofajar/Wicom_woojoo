from joblib import dump
import numpy as np
import pandas as pd
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD
import time
from sklearn.preprocessing import StandardScaler

outliers_fraction = 0.08

data_frame = pd.read_csv("alldata_new.csv")

df_hum1 = np.array([data_frame['humidity1']]).T
df_hum2 = np.array([data_frame['humidity2']]).T
df_temp1 = np.array([data_frame['temperature1']]).T
df_temp2 = np.array([data_frame['temperature2']]).T
df_temp3 = np.array([data_frame['temperature3']]).T
df_waterlevel = np.array([data_frame['waterlevel']]).T


sc_hum1 = StandardScaler().fit(df_hum1)
sc_hum2 = StandardScaler().fit(df_hum2)
sc_temp1 = StandardScaler().fit(df_temp1)
sc_temp2 = StandardScaler().fit(df_temp2)
sc_temp3 = StandardScaler().fit(df_temp3)
sc_waterlevel = StandardScaler().fit(df_waterlevel)

df_sc_hum1 = sc_hum1.transform(df_hum1)
df_sc_hum2 = sc_hum2.transform(df_hum2)
df_sc_temp1 = sc_hum1.transform(df_temp1)
df_sc_temp2 = sc_hum1.transform(df_temp2)
df_sc_temp3 = sc_hum1.transform(df_temp3)
df_sc_waterlevel = sc_hum1.transform(df_waterlevel)


model_hum1 = HBOS(contamination=outliers_fraction)
model_hum2 = HBOS(contamination=outliers_fraction)
model_temp1 = HBOS(contamination=outliers_fraction)
model_temp2 = HBOS(contamination=outliers_fraction)
model_temp3 = HBOS(contamination=outliers_fraction)
model_waterlevel = HBOS(contamination=outliers_fraction)
model_waterleak = HBOS(contamination=outliers_fraction)
model_fire = HBOS(contamination=outliers_fraction)
model_door = HBOS(contamination=outliers_fraction)
'''
model_hum1 = MCD(contamination=outliers_fraction,random_state=42)
model_hum2 = MCD(contamination=outliers_fraction,random_state=42)
model_temp1 = MCD(contamination=outliers_fraction,random_state=42)
model_temp2 = MCD(contamination=outliers_fraction,random_state=42)
model_temp3 = MCD(contamination=outliers_fraction,random_state=42)
model_waterlevel = MCD(contamination=outliers_fraction,random_state=42)
'''

t0 = time.time()
model_hum1.fit(df_sc_hum1)
model_hum2.fit(df_sc_hum2)
model_temp1.fit(df_sc_temp1)
model_temp2.fit(df_sc_temp2)
model_temp3.fit(df_sc_temp3)
model_waterlevel.fit(df_sc_waterlevel)

t1 = time.time()
print(t1-t0)

dump(model_hum1, 'model\HBOS_model_hum1.joblib')
dump(model_hum2, 'model\HBOS_model_hum2.joblib')
dump(model_temp1, 'model\HBOS_model_temp1.joblib')
dump(model_temp2, 'model\HBOS_model_temp2.joblib')
dump(model_temp3, 'model\HBOS_model_temp3.joblib')
dump(model_waterlevel, 'model\HBOS_model_waterlevel.joblib')
'''
dump(model_hum1, 'model\MCD_model_hum1.joblib')
dump(model_hum2, 'model\MCD_model_hum2.joblib')
dump(model_temp1, 'model\MCD_model_temp1.joblib')
dump(model_temp2, 'model\MCD_model_temp2.joblib')
dump(model_temp3, 'model\MCD_model_temp3.joblib')
dump(model_waterlevel, 'model\MCD_model_waterlevel.joblib')
'''
dump(sc_hum1, 'SC\std_scaler_hum1.bin')
dump(sc_hum2, 'SC\std_scaler_hum2.bin')
dump(sc_temp1, 'SC\std_scaler_temp1.bin')
dump(sc_temp2, 'SC\std_scaler_temp2.bin')
dump(sc_temp3, 'SC\std_scaler_temp3.bin')
dump(sc_waterlevel, 'SC\std_scaler_waterlevel.bin')
