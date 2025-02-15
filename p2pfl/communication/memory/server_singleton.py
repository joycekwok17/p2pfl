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

import threading


class ServerSingleton(dict):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(ServerSingleton, cls).__new__(cls)
            cls.termination_event = threading.Event()  # Initialize the threading event
        return cls.instance

    @classmethod
    def reset_instance(cls):
        if hasattr(cls, "instance"):
            cls.termination_event.set()
            del cls.instance

    @classmethod
    def wait_for_termination(cls):
        """
        Blocks until the server is signaled to terminate.
        """
        cls.termination_event.wait()
