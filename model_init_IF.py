from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import time
data_frame = pd.read_csv("data/alldata_new.csv")

df_hum1 = np.array([data_frame['humidity1']]).T
df_hum2 = np.array([data_frame['humidity2']]).T
df_temp1 = np.array([data_frame['temperature1']]).T
df_temp2 = np.array([data_frame['temperature2']]).T
df_temp3 = np.array([data_frame['temperature3']]).T
df_waterlevel = np.array([data_frame['waterlevel']]).T
df_waterleak = np.array([data_frame['leakage']]).T
df_fire = np.array([data_frame['fire']]).T
df_door = np.array([data_frame['isclosed']]).T

estimator = 100
samples = 500
randstate = 42
outlier_fraction = 0.01

model_hum1 = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)
model_hum2 = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)
model_temp1 = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)
model_temp2 = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)
model_temp3 = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)
model_waterlevel = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=0.01)
model_waterleak = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=0.01)
model_fire = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=0.01)
model_door = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=0.01)

t0 = time.time()
model_hum1.fit(df_hum1)
model_hum2.fit(df_hum2)
model_temp1.fit(df_temp1)
model_temp2.fit(df_temp2)
model_temp3.fit(df_temp3)
model_waterlevel.fit(df_waterlevel)
model_waterleak.fit(df_waterleak)
model_fire.fit(df_fire)
model_door.fit(df_door)
t1 = time.time()
print(t1-t0)

dump(model_hum1, 'model\model_hum1.joblib')
dump(model_hum2, 'model\model_hum2.joblib')
dump(model_temp1, 'model\model_temp1.joblib')
dump(model_temp2, 'model\model_temp2.joblib')
dump(model_temp3, 'model\model_temp3.joblib')
dump(model_waterlevel, 'model\model_waterlevel.joblib')
dump(model_waterleak, 'model\model_waterleak.joblib')
dump(model_fire, 'model\model_fire.joblib')
dump(model_door, 'model\model_door.joblib')
