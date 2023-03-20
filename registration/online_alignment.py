from registration.sitkalignment import register_image2, transform_points
from utilities import zmqutils

import json
import os
import time

import threading as tr
import numpy as np

from pathlib import Path
from tifffile import imread, imwrite


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
        self.data_path = data_path

        self.v2p_path = self.data_path.joinpath(r"volts2pix.txt")
        self.p2v_path = self.data_path.joinpath(r"pix2volts.txt")

        self.initialize_from_saved()
        self.comp_id = compname

        self.affine_iterations = 1000
        self.bspline_iterations = 1000
        self.scale_penalty = 100

        self.zmq_input = zmqutils.Subscriber(
            ip=communications_dict["ip"], port=communications_dict["port_sub"]
        )
        self.zmq_output = zmqutils.Pusher(
            ip=communications_dict["ip"], port=communications_dict["port_push"]
        )

        self.running = True

        self.message_reception_tr = tr.Thread(
            target=self.messaging_reception,
            args=(self.zmq_input.socket, self.zmq_output.socket),
        )
        self.message_reception_tr.start()

    def messaging_reception(self, input_sock, output_sock):
        while self.running:
            data = input_sock.recv_multipart()
            data_msg = self.msg_unpacker(data)
            print(data_msg)

            cmd = data_msg["cmd"]
            msg_src = data_msg["source"]
            msg_id = data_msg["id"]

            if cmd == '"set ref images"':
                self.output(
                    output_sock, msg_src, msg_id, cmd, f"processing {cmd}", "pending"
                )
                try:
                    array = np.array(json.loads(data_msg["images"]))
                    # if data_msg["size"][0] == 1:
                    #     array = array[0]
                    self.reference_img = array[0]
                    imwrite(self.data_path.joinpath(r"ref_img.tif"), self.reference_img)
                    self.output(
                        output_sock, msg_src, msg_id, cmd, "ref updated", "complete"
                    )
                except Exception as e:
                    print(f"failed {cmd} because {e}")
                    self.output(output_sock, msg_src, msg_id, cmd, f"{e}", "error")

            elif cmd == '"set target images"':
                self.output(
                    output_sock, msg_src, msg_id, cmd, f"processing {cmd}", "pending"
                )
                try:
                    array = np.array(json.loads(data_msg["images"]))
                    # if data_msg["size"][0] == 1:
                    #     array = array[0]
                    self.target_img = array[0]
                    imwrite(self.data_path.joinpath(r"target_img.tif"), self.target_img)
                    self.output(
                        output_sock, msg_src, msg_id, cmd, "target updated", "complete"
                    )
                except Exception as e:
                    print(f"failed {cmd} because {e}")
                    self.output(output_sock, msg_src, msg_id, cmd, f"{e}", "error")

            elif cmd == '"register"':
                self.output(
                    output_sock, msg_src, msg_id, cmd, f"processing {cmd}", "pending"
                )
                try:
                    print(self.reference_img.shape)
                    print(self.target_img.shape)
                    registered_img = register_image2(
                        self.reference_img,
                        self.target_img,
                        savepath=self.transform_path,
                        scalePenalty=self.scale_penalty,
                        iterations=(self.affine_iterations, self.bspline_iterations),
                    )
                    self.aligned_img = registered_img
                    imwrite(self.data_path.joinpath(r"aligned_img.tif"), registered_img)
                    self.output(
                        output_sock,
                        msg_src,
                        msg_id,
                        cmd,
                        "registration processed",
                        "complete",
                    )

                    try:
                        self.output(
                            output_sock,
                            msg_src,
                            msg_id,
                            cmd,
                            f"processing Pixel/Volt",
                            "pending",
                        )
                        p2v_data = data_msg["T_pxls-to-volts"]
                        v2p_data = data_msg["T_pxls-to-volts"]

                        if os.path.exists(self.v2p_path):
                            os.remove(self.v2p_path)
                        if os.path.exists(self.p2v_path):
                            os.remove(self.p2v_path)

                        with open(self.p2v_path) as file:
                            file.write(p2v_data)
                            file.flush()
                        with open(self.v2p_path) as file:
                            file.write(v2p_data)
                            file.flush()
                        self.output(
                            output_sock,
                            msg_src,
                            msg_id,
                            cmd,
                            f"completed Pixel/Volt",
                            "complete",
                        )
                    except Exception as e:
                        self.output(output_sock, msg_src, msg_id, cmd, f"{e}", "error")

                except Exception as e:
                    print(f"failed {cmd} because {e}")
                    self.output(output_sock, msg_src, msg_id, cmd, f"{e}", "error")

            elif cmd == '"register inverse"':
                self.output(
                    output_sock, msg_src, msg_id, cmd, f"processing {cmd}", "pending"
                )
                try:
                    registered_img = register_image2(
                        self.target_img,
                        self.reference_img,
                        savepath=self.transform_path_INV,
                        scalePenalty=self.scale_penalty,
                        iterations=(self.affine_iterations, self.bspline_iterations),
                    )
                    self.aligned_img_INV = registered_img
                    imwrite(
                        self.data_path.joinpath(r"aligned_img_INV.tif"), registered_img
                    )

                    self.output(
                        output_sock,
                        msg_src,
                        msg_id,
                        cmd,
                        "inverse registration processed",
                        "complete",
                    )
                    try:
                        self.output(
                            output_sock,
                            msg_src,
                            msg_id,
                            cmd,
                            f"processing Pixel/Volt",
                            "pending",
                        )
                        p2v_data = data_msg["T_pxls-to-volts"]
                        v2p_data = data_msg["T_pxls-to-volts"]

                        if os.path.exists(self.v2p_path):
                            os.remove(self.v2p_path)
                        if os.path.exists(self.p2v_path):
                            os.remove(self.p2v_path)

                        with open(self.p2v_path) as file:
                            file.write(p2v_data)
                            file.flush()
                        with open(self.v2p_path) as file:
                            file.write(v2p_data)
                            file.flush()
                        self.output(
                            output_sock,
                            msg_src,
                            msg_id,
                            cmd,
                            f"completed Pixel/Volt",
                            "complete",
                        )
                    except Exception as e:
                        self.output(output_sock, msg_src, msg_id, cmd, f"{e}", "error")

                except Exception as e:
                    print(f"failed {cmd} because {e}")
                    self.output(output_sock, msg_src, msg_id, cmd, f"{e}", "error")

            elif cmd == '"get transformed image"':
                self.output(
                    output_sock, msg_src, msg_id, cmd, f"processing {cmd}", "pending"
                )
                if hasattr(self, "aligned_img"):
                    self.output(
                        output_sock,
                        msg_src,
                        msg_id,
                        cmd,
                        json.dumps(self.aligned_img.tolist()),
                        "complete",
                    )
                else:
                    self.output(
                        output_sock,
                        msg_src,
                        msg_id,
                        cmd,
                        "please run alignment first",
                        "error",
                    )

            elif cmd == '"get inverse transformed image"':
                self.output(
                    output_sock, msg_src, msg_id, cmd, f"processing {cmd}", "pending"
                )
                if hasattr(self, "aligned_img_INV"):
                    self.output(
                        output_sock,
                        msg_src,
                        msg_id,
                        cmd,
                        json.dumps(self.aligned_img_INV.tolist()),
                        "complete",
                    )
                else:
                    self.output(
                        output_sock,
                        msg_src,
                        msg_id,
                        cmd,
                        "please run inverse alignment first",
                        "error",
                    )

            elif cmd == '"points transform"':
                self.output(
                    output_sock, msg_src, msg_id, cmd, f"processing {cmd}", "pending"
                )
                try:
                    coords = eval(data_msg["pnts"])
                    xcoords = [float(xy[0]) for xy in coords]
                    ycoords = [float(xy[1]) for xy in coords]
                    new_coords = [(x, y) for x, y in zip(xcoords, ycoords)]
                    transformed_coords = self.transform_points(new_coords)
                    self.output(
                        output_sock,
                        msg_src,
                        msg_id,
                        cmd,
                        json.dumps(transformed_coords),
                        "complete",
                    )
                except Exception as e:
                    print(f"failed {cmd} because {e}")
                    self.output(output_sock, msg_src, msg_id, cmd, f"{e}", "error")

            elif cmd == '"points inverse transform"':
                self.output(
                    output_sock, msg_src, msg_id, cmd, f"processing {cmd}", "pending"
                )
                try:
                    coords = eval(data_msg["pnts"])
                    xcoords = [float(xy[0]) for xy in coords]
                    ycoords = [float(xy[1]) for xy in coords]
                    new_coords = [(x, y) for x, y in zip(xcoords, ycoords)]
                    transformed_coords = self.transform_points_INV(new_coords)
                    self.output(
                        output_sock,
                        msg_src,
                        msg_id,
                        cmd,
                        json.dumps(transformed_coords),
                        "complete",
                    )
                except Exception as e:
                    print(f"failed {cmd} because {e}")
                    self.output(output_sock, msg_src, msg_id, cmd, f"{e}", "error")

            elif cmd == '"spawn new ip"':
                self.output(
                    output_sock, msg_src, msg_id, cmd, f"processing {cmd}", "pending"
                )
                try:
                    new_ip = data_msg["ip"]
                    new_sub_port = data_msg["output"]
                    new_push_port = data_msg["input"]

                    zmq_input = zmqutils.Subscriber(ip=new_ip, port=new_sub_port)
                    zmq_output = zmqutils.Pusher(ip=new_ip, port=new_push_port)

                    self.new_msg_tr = tr.Thread(
                        target=self.messaging_reception,
                        args=(zmq_input.socket, zmq_output.socket),
                    )
                    self.new_msg_tr.start()
                    self.output(
                        output_sock,
                        msg_src,
                        msg_id,
                        cmd,
                        f"new {cmd} created",
                        "complete",
                    )
                except Exception as e:
                    print(f"failed {cmd} because {e}")
                    self.output(output_sock, msg_src, msg_id, cmd, f"{e}", "error")

            elif cmd == "get T_volts-to-pxls":
                self.output(
                    output_sock, msg_src, msg_id, cmd, f"processing {cmd}", "pending"
                )
                try:
                    with open(self.v2p_path) as file:
                        data = file.read()
                    self.output(
                        output_sock,
                        msg_src,
                        msg_id,
                        cmd,
                        data,
                        "complete",
                    )
                except Exception as e:
                    self.output(output_sock, msg_src, msg_id, cmd, f"{e}", "error")

            # elif cmd == "set T_volts-to-pxls":
            #     self.output(
            #         output_sock, msg_src, msg_id, cmd, f"processing {cmd}", "pending"
            #     )
            #     try:
            #         if os.path.exists(self.v2p_path):
            #             os.remove(self.v2p_path)
            #
            #         with open(self.v2p_path) as file:
            #             file.write(data_msg["data"])
            #             file.flush()
            #         self.output(
            #             output_sock,
            #             msg_src,
            #             msg_id,
            #             cmd,
            #             data,
            #             "complete",
            #         )
            #     except Exception as e:
            #         self.output(output_sock, msg_src, msg_id, cmd, f"{e}", "error")

            elif cmd == "get T_pxls-to-volts":
                self.output(
                    output_sock, msg_src, msg_id, cmd, f"processing {cmd}", "pending"
                )
                try:
                    with open(self.p2v_path) as file:
                        data = file.read()
                    self.output(
                        output_sock,
                        msg_src,
                        msg_id,
                        cmd,
                        data,
                        "complete",
                    )
                except Exception as e:
                    self.output(output_sock, msg_src, msg_id, cmd, f"{e}", "error")

            # elif cmd == "set T_pxls-to-volts":
            #     self.output(
            #         output_sock, msg_src, msg_id, cmd, f"processing {cmd}", "pending"
            #     )
            #     try:
            #         if os.path.exists(self.p2v_path):
            #             os.remove(self.p2v_path)
            #
            #         with open(self.p2v_path) as file:
            #             file.write(data_msg["data"])
            #             file.flush()
            #         self.output(
            #             output_sock,
            #             msg_src,
            #             msg_id,
            #             cmd,
            #             data,
            #             "complete",
            #         )
            #     except Exception as e:
            #         self.output(output_sock, msg_src, msg_id, cmd, f"{e}", "error")
            else:
                print(f"{cmd} not understood")
                self.output(
                    output_sock,
                    msg_src,
                    msg_id,
                    cmd,
                    "cmd not understood",
                    "error",
                )

    def transform_points(self, points):
        transformed_points = transform_points(
            self.transform_path, points, floating=True
        )
        return transformed_points

    def transform_points_INV(self, points):
        transformed_points = transform_points(
            self.transform_path_INV, points, floating=True
        )
        return transformed_points

    def initialize_from_saved(self):
        try:
            self.target_img = imread(self.data_path.joinpath(r"target_img.tif"))
        except:
            print("no target found")

        try:
            self.reference_img = imread(self.data_path.joinpath(r"ref_img.tif"))
        except:
            print("no ref found")

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

    def output(
        self,
        output_sock,
        msg_src,
        msg_id,
        msg_type,
        msg_data,
        process_status="complete",
    ):
        output_sock.send_multipart(
            [
                "dest".encode(),
                msg_src.encode(),
                "id".encode(),
                msg_id.encode(),
                "time".encode(),
                str(time.time()).encode(),
                "cmd".encode(),
                msg_type[1:-1].encode(),
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


if __name__ == "__main__":
    # used_comms = {
    #     "ip": "tcp://10.196.144.133:",
    #     "port_sub": "5555",
    #     "port_push": "5556",
    # }
    # input_path = r"D:\Data\alignment_sample"
    '''
    python -m registration.online_alignment --filename="D:/Data/alignment_sample" --ip="tcp://10.196.144.133:" --port_sub="5555" --port_push="5556"
    '''
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--filename")
    parser.add_argument("--ip")
    parser.add_argument("--port_sub")
    parser.add_argument("--port_push")

    args = parser.parse_args()
    try:
        input_path = Path(args.filename)

        used_comms = {
            "ip": args.ip,
            "port_sub": args.port_sub,
            "port_push": args.port_push,
        }
    except TypeError:
        # if user doesnt put stuff in
        used_comms = {
            "ip": "tcp://localhost:",
            # "ip": "tcp://10.196.144.133:",
            "port_sub": "5555",
            "port_push": "5556",
        }
        input_path = Path(os.path.expanduser(r'~\Documents\registration'))
    OA = OnlineAlign(communications_dict=used_comms, data_path=input_path)
