FROM python:3.8-slim
WORKDIR /root/

RUN python3 -m pip install pyiotown
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install pyod
RUN python3 -m pip install joblib
RUN python3 -m pip install numpy
RUN python3 -m pip install pandas

COPY ./ .

CMD python3 -u postproc_IF.py
