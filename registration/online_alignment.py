from registration.sitkalignment import register_image2, transform_points
from utilities import zmqutils

import json
import os
import time

import threading as tr
import numpy as np

from pathlib import Path
from tifffile import imread, imsave

# r'D:\Data\alignment_sample


class OnlineAlign:
    def __init__(self, communications_dict, data_path, compname="regComp"):
        """

        :param comms: dictionary:
                                 {
                                 ip_address
                                 port_sub
                                 port_push
                                 }
        :param savepath: homebase for alignments and such
        :param defaultSize: default image size
        """
        self.data_path = Path(data_path)
        self.initialize_from_saved()
        self.comp_id = compname

        self.affine_iterations = 1000
        self.bspline_iterations = 1000
        self.scale_penalty = 100

        self.zmq_input = zmqutils.Subscriber(
            ip=communications_dict["ip"], port=communications_dict["port_sub"]
        )
        self.zmq_output = zmqutils.Pusher(
            ip=communications_dict["ip"], port=communications_dict["port_sub"]
        )

        self.running = True

        self.message_reception_tr = tr.Thread(target=self.messaging_reception)
        self.message_reception_tr.start()

    def messaging_reception(self):
        while self.running:
            data = self.zmq_input.socket.recv_multipart()
            data_msg = self.msg_unpacker(data)
            print(data_msg)

            cmd = data_msg["cmd"]
            msg_src = data_msg["source"]
            msg_id = data_msg["id"]

            if cmd == "images_ref":
                array = np.array(json.loads(data_msg["images"]))
                if data_msg["size"][0] == 1:
                    array = array[0]
                self.reference_img = array
                imsave(self.data_path.joinpath(r"ref_img.tif"), self.reference_img)
                self.output(msg_src, msg_id, cmd, "ref updated", "complete")

            elif cmd == "images_target":
                array = np.array(json.loads(data_msg["images"]))
                if data_msg["size"][0] == 1:
                    array = array[0]
                self.target_img = array
                imsave(self.data_path.joinpath(r"target_img.tif"), self.target_img)
                self.output(msg_src, msg_id, cmd, "target updated", "complete")

            elif cmd == "register":
                self.output(msg_src, msg_id, cmd, "processing registration", "pending")
                registered_img = register_image2(
                    self.reference_img,
                    self.target_img,
                    savepath=self.data_path,
                    scalePenalty=self.scale_penalty,
                    iterations=(self.affine_iterations, self.bspline_iterations),
                )
                self.aligned_img = registered_img
                imsave(self.data_path.joinpath(r"aligned_img.tif"), registered_img)
                self.output(msg_src, msg_id, cmd, "registration processed", "complete")

            elif cmd == "register inverse":
                self.output(
                    msg_src, msg_id, cmd, "processing inverse registration", "pending"
                )
                registered_img = register_image2(
                    self.target_img,
                    self.reference_img,
                    savepath=self.data_path,
                    scalePenalty=self.scale_penalty,
                    iterations=(self.affine_iterations, self.bspline_iterations),
                )
                self.aligned_img_INV = registered_img
                imsave(self.data_path.joinpath(r"aligned_img_INV.tif"), registered_img)
                self.output(
                    msg_src, msg_id, cmd, "inverse registration processed", "complete"
                )

            elif cmd == "image transform":
                self.output(
                    msg_src,
                    msg_id,
                    cmd,
                    json.dumps(self.aligned_img.tolist()),
                    "complete",
                )

            elif cmd == "image inverse transform":
                self.output(
                    msg_src,
                    msg_id,
                    cmd,
                    json.dumps(self.aligned_img_INV.tolist()),
                    "complete",
                )

            elif cmd == "points transform":
                self.output(
                    msg_src, msg_id, cmd, "processing points transform", "pending"
                )
                coords = data_msg["coords"]
                xcoords = [float(i) for i in coords[0]]
                ycoords = [float(i) for i in coords[1]]
                new_coords = [(x, y) for x, y in zip(xcoords, ycoords)]
                transformed_coords = self.transform_points(new_coords)
                self.output(
                    msg_src, msg_id, cmd, json.dumps(transformed_coords), "complete"
                )

            elif cmd == "points inverse transform":
                self.output(
                    msg_src,
                    msg_id,
                    cmd,
                    "processing inverse points transform",
                    "pending",
                )
                coords = data_msg["coords"]
                xcoords = [float(i) for i in coords[0]]
                ycoords = [float(i) for i in coords[1]]
                new_coords = [(x, y) for x, y in zip(xcoords, ycoords)]
                transformed_coords = self.transform_points(new_coords)
                self.output(
                    msg_src, msg_id, cmd, json.dumps(transformed_coords), "complete"
                )

            else:
                print(f"{cmd} not understood")

    def transform_points(self, points):
        transformed_points = transform_points(self.transform_path, points)
        return transformed_points

    def transform_points_INV(self, points):
        transformed_points = transform_points(self.transform_path_INV, points)
        return transformed_points

    def initialize_from_saved(self):
        self.target_img = imread(self.data_path.joinpath(r"target_img.tif"))
        self.reference_img = imread(self.data_path.joinpath(r"ref_img.tif"))
        self.transform_path = self.data_path.joinpath("transform")
        self.transform_path_INV = self.data_path.joinpath("transform_INV")
        try:
            self.aligned_img_INV = imread(
                self.data_path.joinpath(r"aligned_img_INV.tif")
            )
        except:
            print("aligned inverse image unavailable")

        try:
            self.aligned_img = imread(self.data_path.joinpath(r"aligned_img.tif"))
        except:
            print("aligned image unavailable")

        if not os.path.exists(self.transform_path):
            os.mkdir(self.transform_path)
        if not os.path.exists(self.transform_path_INV):
            os.mkdir(self.transform_path_INV)

    def output(self, msg_src, msg_id, msg_type, msg_data, process_status="complete"):
        self.zmq_output.socket.send_multipart(
            [
                "dest".encode(),
                msg_src.encode(),
                "id".encode(),
                msg_id.encode(),
                "time".encode(),
                str(time.time()).encode(),
                "cmd".encode(),
                msg_type.encode(),
                "data".encode(),
                msg_data.encode(),
                "status".encode(),
                process_status.encode(),
                "source".encode(),
                self.comp_id.encode(),
            ]
        )
        print(
            [
                "dest",
                msg_src,
                "id",
                msg_id,
                "time",
                str(time.time()),
                "cmd",
                msg_type,
                "data",
                msg_data,
            ]
        )

    @staticmethod
    def msg_unpacker(msg):
        keys = msg[::2]
        vals = msg[1::2]

        msg_dict = {}
        for k, v in zip(keys, vals):
            msg_dict[k.decode()] = v.decode()
        return msg_dict
