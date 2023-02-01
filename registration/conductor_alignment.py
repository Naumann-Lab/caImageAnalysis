import zmq
import json

import threading as tr
import numpy as np

from datetime import datetime as dt

## local imports
try:
    from registration import register_image2, transform_points
except:
    pass
try:
    from registration.sitkalignment import register_image2, transform_points
except:
    pass


class ConductorAlignment:
    def __init__(self, savepath=None, defaultSize=512, inputPort=5592, outputPort=5593):

        self.savepath = savepath
        self.zmq_input_port = str(inputPort)
        self.zmq_input = Subscriber(self.zmq_input_port)

        self.zmq_output_port = str(outputPort)
        self.zmq_output = Publisher(self.zmq_output_port)

        self.default_size = (defaultSize, defaultSize)

        self.running = True
        self.images = {}
        self.receiving_thread = tr.Thread(target=self.labViewImgReceiver)
        self.receiving_thread.start()

    def labViewImgReceiver(self):
        while self.running:
            data = self.zmq_input.socket.recv()
            print(data)
            msg_parts = data.decode("utf-8").split(";")
            tag = msg_parts[0]
            if tag == "target_img":
                print("loading target image")
                dateString = str(msg_parts[0]).split(" ")[2]
                timestamp = dt.strptime(dateString, "%H:%M:%S.%f").time()
                array = np.array(json.loads(msg_parts[1]))[:, 32:]
                self.images["target"] = array
            elif tag == "reference_img":
                print("loading reference image")
                dateString = str(msg_parts[0]).split(" ")[2]
                timestamp = dt.strptime(dateString, "%H:%M:%S.%f").time()
                array = np.array(json.loads(msg_parts[1]))[:, 32:]
                self.images["reference"] = array
                self.default_size = (
                    self.images["reference"].shape[0],
                    self.images["reference"].shape[1],
                )

            elif tag == "run_alignment":
                print("attempting alignment")
                if (
                    not "reference" in self.images.keys()
                    and not "target" in self.images.keys()
                ):
                    self.zmq_output.socket.send(f"ERROR: failed to load both images")
                    print(f"ERROR: failed to load both images")
                else:
                    self.runAlignment()
            elif tag == "points":
                print("pointing")
                self._points = msg_parts[1].split(":")

                self.points = []
                for pt in self._points:
                    x, y = [float(i) for i in pt.split(",")]
                    self.points.append((x, y))

                print(self.points)
                self.alignPoints()
            elif tag == "sizeChange":
                self.default_size = msg_parts[0].split(":")
            else:
                print(f"{tag} not understood")

    def runAlignment(self):
        register_image2(
            self.images["reference"],
            self.images["target"],
            savepath=self.savepath,
            iterations=(1500, 1500),
        )

    def alignPoints(self):

        self.conv_points = transform_points(self.savepath, self.points)
        print([f"{pt};" for pt in self.conv_points])
        outputStr = ""
        for point in self.conv_points:
            outputStr += str(point[0]) + "," + str(point[1]) + ":"
        self.zmq_output.socket.send(outputStr.encode())


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
