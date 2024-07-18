#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import sys
import time

from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import (
    MnistFederatedDM,
)
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import argparse

"""
Example of a P2PFL MNIST node using a MLP model and a MnistFederatedDM. 
This node will be connected to node1 and then, the federated learning process will start.
"""

def __get_args():
    parser = argparse.ArgumentParser(description="P2PFL MNIST node using a MLP model and a MnistFederatedDM.")
    parser.add_argument("self_address", type=str, help="self ip address and the port number -> ip addr: portnumber.")
    parser.add_argument("other_address", type=str, help="The ip address and the port number of the other node -> ip addr: portnumber.")
    return parser.parse_args()

def node2(self_address, other_address):
    node = Node(
        MLP(),
        MnistFederatedDM(sub_id=1, number_sub=2),
        # port=int(sys.argv[1]),
        address= self_address,    # self address would be "192.168.1.1:50051"
    )
    node.start()
    parsed_other_address = other_address.split(":")
    # node.connect(f"127.0.0.1:{port}")
    node.connect(f"{parsed_other_address[0]}:{parsed_other_address[1]}")
    time.sleep(4)

    node.set_start_learning(rounds=2, epochs=1)

    # Wait 4 results

    while True:
        time.sleep(1)

        if node.round is None:
            break

    node.stop()

if __name__ == "__main__":
    # Get arguments
    args = __get_args()

    # Run node2
    node2(args.self_address, args.other_address)

