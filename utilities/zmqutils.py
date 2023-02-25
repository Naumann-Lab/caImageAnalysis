import zmq


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


class Pusher:
    """
    Subscriber wrapper class for zmq.
    Default topic is every topic ("").
    """

    def __init__(self, port="1234", ip=None):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        if ip is not None:
            self.socket.connect(ip + str(self.port))
        else:
            self.socket.connect("tcp://localhost:" + str(self.port))

    def kill(self):
        self.socket.close()
        self.context.term()
