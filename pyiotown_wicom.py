import sys
import requests
import json
import threading
from urllib.parse import urlparse
import paho.mqtt.client as mqtt
import json

def storage(url, token, nid, date_from , date_to, lastKey="", verify=True, timeout=60):
	header = {'Accept':'application/json','token':token}
	apiaddr = url + "/api/v1.0/storage?nid=" + nid + "&from=" + date_from + "&to=" + date_to
	if lastKey != "":
		apiaddr = url + "/api/v1.0/storage?nid=" + nid + "&from=" + date_from + "&to=" + date_to + "&lastKey=" + lastKey
	try:
		print(apiaddr)
		r = requests.get(apiaddr,headers=header, verify=verify, timeout=timeout)
		if r.status_code == 200:
			return r.json()
		else:
			print(r.content)
			return None
	except Exception as e:
		print(e)
		return None

def gateways_connect(url, token, gateway_id, verify=True, timeout=60): 
    #This API allows you to get a list of the child nodes of a specific gateway that are associated with IoT.own.
    header = {'Accept':'application/json','token':token}
    apiaddr = url + "/api/v1.0/gateway/{gateway_id}/connected-nodes"
    try:
        print(apiaddr)
        r = requests.get(apiaddr,headers=header, verify=False, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        else:
            print(r.content)
            return None
    except Exception as e:
        print(e)
        return None

def gateways(url, token, verify=True, timeout=60):
	header = {'Accept':'application/json','token':token}
	apiaddr = url + "/api/v1.0/gateways"
	try:
		print(apiaddr)
		r = requests.get(apiaddr,headers=header, verify=False, timeout=timeout)
		if r.status_code == 200:
			return r.json()
		else:
			print(r.content)
			return None
	except Exception as e:
		print(e)
		return None

def nodes(url, token, verify=True, timeout=60):
	header = {'Accept':'application/json','token':token}
	apiaddr = url + "/api/v1.0/nodes"
	try:
		print(apiaddr)
		r = requests.get(apiaddr,headers=header, verify=False, timeout=timeout)
		if r.status_code == 200:
			return r.json()
		else:
			print(r.content)
			return None
	except Exception as e:
		print(e)
		return None

def data(url, token, nid, data, upload="", verify=True, timeout=60):
    typenum = "2" # 2 static 
    apiaddr = url + "/api/v1.0/data"
    if upload == "":
        header = {'Accept':'application/json', 'token':token } 
        payload = { "type" : typenum, "nid" : nid, "data": data }
        try:
            r = requests.post(apiaddr, json=payload, headers=header, verify=verify, timeout=timeout)
            if r.status_code == 200:
                return True
            else:
                print(r.content)
                return False
        except Exception as e:
            print(e)
            return False
    else:
        header = {'Accept':'application/json', 'token':token } 
        payload = { "type" : typenum, "nid" : nid, "meta": json.dumps(data) }
        try:
            r = requests.post(apiaddr, data=payload, headers=header, verify=verify, timeout=timeout, files=upload)
            if r.status_code == 200:
                return True
            else:
                print(r.content)
                return False
        except Exception as e:
            print(e)
            return False

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connect OK! Subscribe Start")
    else:
        print("Bad connection Reason : {}".format(rc))

def post_files(result, url, token, verify=True, timeout=60):
    if 'data' not in result.keys():
        return result
    
    for key in result['data'].keys():
        if type(result['data'][key]) is dict:
            resultkey = result['data'][key].keys()
            if ('raw' in resultkey) and ( 'file_type' in resultkey) :
                header = {'Accept':'application/json', 'token':token }
                upload = { key + "file": result['data'][key]['raw'] }
                try:
                    r = requests.post( url + "/api/v1.0/file", headers=header, verify=verify, timeout=timeout, files=upload )
                    if r.status_code == 200:
                        del result['data'][key]['raw']
                        result['data'][key]['file_id'] = r.json()["files"][0]["file_id"]
                        result['data'][key]['file_ext'] = r.json()["files"][0]["file_ext"]
                        result['data'][key]['file_size'] = r.json()["files"][0]["file_size"]
                    else:
                        print("[ Error ] while send Files to IoT.own. check file format ['raw, file_type]")
                        print(r.content)
                except Exception as e:
                    print(e)
            # post post process apply.
    return result

def on_message(client, userdata, msg):
    data = json.loads((msg.payload).decode('utf-8'))
    try:
        result = userdata['func'](data)
    except Exception as e:
        print(f'Error on calling the user-defined function', file=sys.stderr)
        print(e, file=sys.stderr)
        client.publish('iotown/proc-done', msg.payload, 1)
        return
    
    if type(result) is dict and 'data' in result.keys():
        if 'token' in userdata.keys():
            result = post_files(result, userdata['url'], userdata['token'])
        client.publish('iotown/proc-done', json.dumps(result), 1)
    elif result is None:
        print(f"Discard the message")
    else:
        print(f"CALLBACK FUNCTION TYPE ERROR {type(result)} must [ dict ]", file=sys.stderr)
        client.publish('iotown/proc-done', msg.payload, 1)

    
def updateExpire(url, token, name, verify=True, timeout=60):
    apiaddr = url + "/api/v1.0/pp/proc"
    header = {'Accept':'application/json', 'token':token}
    payload = { 'name' : name}
    try:
        r = requests.post(apiaddr, json=payload, headers=header, verify=False, timeout=timeout)
        if r.status_code == 200 or r.status_code == 403:
            print("update Expire Success")
        else:
            print("update Expire Fail! reason:",r)
    except Exception as e:
        print("update Expire Fail! reason:", e)
    timer = threading.Timer(60, updateExpire,[url,token,name])
    timer.start()

def getTopic(url, token, name, verify=True, timeout=60):
    apiaddr = url + "/api/v1.0/pp/proc"
    header = {'Accept':'application/json', 'token':token}
    payload = {'name':name}
    print('here')    
    try:
        r = requests.post(apiaddr, json=payload, headers=header, verify=False, timeout=timeout)
        print(r)
        if r.status_code == 200:
            print("Get Topic From IoT.own Success")
            return json.loads((r.content).decode('utf-8'))['topic']
        elif r.status_code == 403:
            print("process already in use. please restart after 1 minute later.")
            return json.loads((r.content).decode('utf-8'))['topic']
        else:
            print(r)
            return None
    except Exception as e:
        print(e)
        return None

def postprocess(url, name, func, username, pw, port=8883):
    # get Topic From IoTown
    topic = getTopic(url, pw, name)

    if topic == None:
        raise Exception("IoT.own returned none")

    try:
        group = topic.split('/')[2]
    except Exception as e:
        raise Exception(f"Invalid topic {topic}")
    
    updateExpire(url, pw, name)
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.username_pw_set(username, pw)
    client.user_data_set({
        "url": url,
        "token": pw,
        "func": func,
        "group": group
    })
    mqtt_server = urlparse(url).hostname
    print(f"connect to {mqtt_server}:{port}")
    client.tls_set()
    client.tls_insecure_set(True)
    client.connect(mqtt_server, port=port)
    client.subscribe(topic, 1)
    client.loop_forever()