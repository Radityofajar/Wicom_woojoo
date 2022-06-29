import socket
import time
import json
from pyiotown import post
import warnings
warnings.filterwarnings("ignore")

#Iot.own network server parameter
url = "https://town.coxlab.kr/"
token = "c34859e08fa526f642881820d5108ccd475d5b58efbc8b4a5b89fd93366fe1d1"
nodeid = "LW111100001111BBCD"

#server for receive data from sensor
host = '192.168.0.20' 
port = 80
addr = host, port
BF = 1000
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(addr)
server.listen(10000)

def receive_data(server,buffer_size=BF):
    clientsocket, address = server.accept()
    msg_test = clientsocket.recv(4096)
    msg_test = msg_test.decode('utf-8')
    return msg_test

if __name__ == '__main__':
	while True:
		time.sleep(0.8)
		payload = receive_data(server)
		test = payload.split()
		print(len(test))
		if len(test) >= 11:
			r = post.data(url,token,nodeid,payload)
			print(r)
		else:
			payload = ('Data is not complete')
			r = post.data(url,token,nodeid,payload)
			print('data is not complete')
