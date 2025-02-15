#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/federated_learning_p2p).
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

from typing import List, Union, Dict, Any
from datetime import datetime

from p2pfl.commands.command import Command
from p2pfl.communication.gossiper import Gossiper
from p2pfl.communication.memory.memory_neighbors import InMemoryNeighbors
from p2pfl.communication.memory.server_singleton import ServerSingleton
from p2pfl.management.logger import logger


class InMemoryServer:
    """
    Implementation of the server side of an in-memory communication protocol.
    """

    __server = None  # Class variable to hold the singleton instance

    ####
    # Init
    ####

    def __init__(
        self,
        addr: str,
        gossiper: Gossiper,
        neighbors: InMemoryNeighbors,
        commands: List[Command] = [],
    ) -> None:
        # Message handlers
        self.__commands = {c.get_name(): c for c in commands}

        # Address
        self.addr = addr

        # Gossiper
        self.__gossiper = gossiper

        # Neighbors
        self.__neighbors = neighbors

        # Server
        self.__server = ServerSingleton()

    ####
    # Management
    ####

    def start(self, wait: bool = False) -> None:
        """
        Starts the in-memory server.
        """
        self.__server[self.addr] = self
        logger.info(self.addr, f"InMemoryServer started at {self.addr}")

    def stop(self) -> None:
        """
        Stops the in-memory server.
        """
        self.__server = ServerSingleton.reset_instance()
        logger.info(self.addr, f"InMemoryServer stopped at {self.addr}")

    def wait_for_termination(self) -> None:
        """
        Waits for termination.
        """
        ServerSingleton.wait_for_termination()

    ####
    # In-Memory Services
    ####

    def handshake(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        In-memory service. It is called when a node connects to another.
        """
        if self.__neighbors.add(request["addr"], non_direct=False, handshake_msg=False):
            return {}
        else:
            return {"error": "Cannot add the node (duplicated or wrong direction)"}

    def disconnect(self, request: Dict[str, Any]) -> None:
        """
        In-memory service. It is called when a node disconnects from another.
        """
        self.__neighbors.remove(request["addr"], disconnect_msg=False)

    def send_message(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        In-memory service. It is called when a node sends a message to another.
        """
        # If not processed
        if self.__gossiper.check_and_set_processed(request["hash"]):
            logger.debug(
                self.addr,
                f"Received message from {request['source']} > {request['cmd']} {request['args']}",
            )
            # Gossip
            if request["ttl"] > 1:
                # Update ttl and gossip
                request["ttl"] -= 1
                pending_neis = [
                    n
                    for n in self.__neighbors.get_all(only_direct=True).keys()
                    if n != request["source"]
                ]
                self.__gossiper.add_message(request, pending_neis)

            # Process message
            if request["cmd"] in self.__commands.keys():
                try:
                    self.__commands[request["cmd"]].execute(
                        request["source"], request["round"], *request["args"]
                    )
                except Exception as e:
                    error_text = (
                        f"Error while processing command: {request['cmd']} {request['args']}: {e}"
                    )
                    logger.error(self.addr, error_text)
                    return {"error": error_text}
            else:
                logger.error(
                    self.addr, f"Unknown command: {request['cmd']} from {request['source']}"
                )
                return {"error": f"Unknown command: {request['cmd']}"}

        return {}

    def send_weights(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        In-memory service. It is called when a node sends weights to another.
        """
        # Process message
        if request["cmd"] in self.__commands.keys():
            try:
                self.__commands[request["cmd"]].execute(
                    request["source"],
                    request["round"],
                    request["weights"],
                    request["contributors"],
                    request["weight"],
                )
            except Exception as e:
                error_text = f"Error while processing model: {request['cmd']}: {e}"
                print(error_text)
                return {"error": error_text}
        else:
            print(f"Unknown command: {request['cmd']} from {request['source']}")
            return {"error": f"Unknown command: {request['cmd']}"}
        return {}

    ####
    # Commands
    ####

    def add_command(self, cmds: Union["Command", List["Command"]]) -> None:
        """
        Adds a command.
        """
        if isinstance(cmds, list):
            for cmd in cmds:
                self.__commands[cmd.get_name()] = cmd
        elif isinstance(cmds, Command):
            self.__commands[cmds.get_name()] = cmds
        else:
            raise Exception("Command not valid")
