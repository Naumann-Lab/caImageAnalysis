import zmq
import json

import threading as tr
import numpy as np

from datetime import datetime as dt

## local imports
from sitkalignment import register_image2, return_conv_pt


class ConductorAlignment:
    def __init__(self, savepath=None, inputPort=5592, outputPort=5593):

        self.savepath = savepath
        self.zmq_input_port = str(inputPort)
        self.zmq_input = Subscriber(self.zmq_input_port)

        self.zmq_output_port = str(outputPort)
        self.zmq_output = Publisher(self.zmq_output_port)

        self.images = {}
        self.receiving_thread = tr.Thread(target=self.labViewImgReceiver)
        self.receiving_thread.start()
        self.running = True

    def labViewImgReceiver(self):
        while self.running:
            data = self.zmq_input.socket.recv()
            msg_parts = [part.strip() for part in data.split(b": ", 1)]
            tag = msg_parts[0].split(b' ')[0]
            if tag == "target_img":
                print('loading target image')
                dateString = str(msg_parts[0]).split(" ")[2]
                timestamp = dt.strptime(dateString, "%H:%M:%S.%f").time()
                array = np.array(json.loads(msg_parts[1]))[
                        :, 32:
                        ]
                self.images['target'] = array
            elif tag == "reference_img":
                print('loading reference image')
                dateString = str(msg_parts[0]).split(" ")[2]
                timestamp = dt.strptime(dateString, "%H:%M:%S.%f").time()
                array = np.array(json.loads(msg_parts[1]))[
                        :, 32:
                        ]
                self.images['reference'] = array
            elif tag == "run_alignment":
                print('attempting alignment')
                if not "reference" in self.images.keys() and not "target" in self.images.keys():
                    self.zmq_output.socket.send(f"ERROR: failed to load both images")
                    print(f"ERROR: failed to load both images")
                else:
                    self.runAlignment()
            elif tag == "points":
                print('pointing')
                self.points = msg_parts[0].split(':')
                self.alignPoints()
            else:
                print(f'{tag} not understood')

    def runAlignment(self):
        register_image2(self.images['reference'],
                        self.images['target'],
                        savepath=self.savepath,
                        iterations=(1500, 1500))

    def alignPoints(self):
        self.imgSize = self.images['reference'].shape
        self.conv_points = []
        for point in self.points:
            x, y = return_conv_pt(point[1],
                           point[0],
                           self.savepath,
                           size1=self.imgSize[0],
                           size2=self.imgSize[1])
            self.conv_points.append((x,y))
        self.zmq_output.socket.send([f'{pt};' for pt in self.conv_points])

class Subscriber:
    """
    Subscriber wrapper class for zmq.
    Default topic is every topic ("").
    """

    def __init__(self, port="1234", topic="", ip=None):
        self.port = port
        self.topic = topic
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        if ip is not None:
            self.socket.connect(ip + str(self.port))
        else:
            self.socket.connect("tcp://localhost:" + str(self.port))
        self.socket.subscribe(self.topic)

    def kill(self):
        self.socket.close()
        self.context.term()


class Publisher:
    """
    Publisher wrapper class for zmq.
    """

    def __init__(self, port="1234"):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:" + self.port)

    def kill(self):
        self.socket.close()
        self.context.term()
