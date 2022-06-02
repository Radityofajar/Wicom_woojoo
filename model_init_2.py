from joblib import dump
import numpy as np
import pandas as pd
from pyod.models.hbos import HBOS
#from pyod.models.mcd import MCD
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
df_waterleak = np.array([data_frame['leakage']]).T
df_fire = np.array([data_frame['fire']]).T
df_door = np.array([data_frame['isclosed']]).T

sc_hum1 = StandardScaler().fit_transform(df_hum1)
sc_hum2 = StandardScaler().fit_transform(df_hum2)
sc_temp1 = StandardScaler().fit_transform(df_temp1)
sc_temp2 = StandardScaler().fit_transform(df_temp2)
sc_temp3 = StandardScaler().fit_transform(df_temp3)
sc_waterlevel = StandardScaler().fit_transform(df_waterlevel)
sc_waterleak = StandardScaler().fit_transform(df_waterleak)
sc_fire = StandardScaler().fit_transform(df_fire)
sc_door = StandardScaler().fit_transform(df_door)

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
model_waterleak = MCD(contamination=outliers_fraction,random_state=42)
model_fire = MCD(contamination=outliers_fraction,random_state=42)
model_door = MCD(contamination=outliers_fraction,random_state=42)
'''
t0 = time.time()
model_hum1.fit(sc_hum1)
model_hum2.fit(sc_hum2)
model_temp1.fit(sc_temp1)
model_temp2.fit(sc_temp2)
model_temp3.fit(sc_temp3)
model_waterlevel.fit(sc_waterlevel)
model_waterleak.fit(sc_waterleak)
model_fire.fit(sc_fire)
model_door.fit(sc_door)
t1 = time.time()
print(t1-t0)

dump(model_hum1, 'model\HBOS_model_hum1.joblib')
dump(model_hum2, 'model\HBOS_model_hum2.joblib')
dump(model_temp1, 'model\HBOS_model_temp1.joblib')
dump(model_temp2, 'model\HBOS_model_temp2.joblib')
dump(model_temp3, 'model\HBOS_model_temp3.joblib')
dump(model_waterlevel, 'model\HBOS_model_waterlevel.joblib')
dump(model_waterleak, 'model\HBOS_model_waterleak.joblib')
dump(model_fire, 'model\HBOS_model_fire.joblib')
dump(model_door, 'model\HBOS_model_door.joblib')

dump(sc_hum1, 'SC\std_scaler_hum1.bin')
dump(sc_hum2, 'SC\std_scaler_hum2.bin')
dump(sc_temp1, 'SC\std_scaler_temp1.bin')
dump(sc_temp2, 'SC\std_scaler_temp2.bin')
dump(sc_temp3, 'SC\std_scaler_temp3.bin')
dump(sc_waterlevel, 'SC\std_scaler_waterlevel.bin')
dump(sc_waterleak, 'SC\std_scaler_waterleak.bin')
dump(sc_fire, 'SC\std_scaler_fire.bin')
dump(sc_door, 'SC\std_scaler_door.bin')
