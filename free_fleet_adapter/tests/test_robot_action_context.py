#!/usr/bin/env python3

# Copyright 2026 AIT Austrian Institute of Technology GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from free_fleet_adapter.action import RobotActionContext
from free_fleet_adapter.ros2_action_client import Ros2ActionClient

from tests.fake_ros2_action_endpoint import (
    FakeCodec,
    FakeNode,
    FakeZenohSession,
)


def make_context(create_ros2_action_client=None):
    return RobotActionContext(
        node=FakeNode(),
        robot_name='robogreen',
        update_handle=None,
        fleet_config=None,
        action_config={'actions': []},
        get_battery_soc=lambda: 1.0,
        get_map_name=lambda: 'L1',
        get_pose=lambda: [0.0, 0.0, 0.0],
        create_ros2_action_client=create_ros2_action_client
    )


class TestRobotActionContext(unittest.TestCase):
    """A plugin reaches the robot's ROS 2 actions through its context."""

    def test_a_plugin_obtains_a_client_for_its_own_endpoint(self):
        node = FakeNode()
        session = FakeZenohSession()
        made = []

        def _make_client(action_name, codec):
            client = Ros2ActionClient(
                node, session, 'robogreen', action_name, codec, 1.0)
            made.append(client)
            return client

        context = make_context(_make_client)
        codec = FakeCodec()

        client = context.create_ros2_action_client('sweep_the_field', codec)

        assert client is made[0]
        assert client.action_name == 'sweep_the_field'
        assert client.codec is codec

    def test_the_endpoint_is_named_when_no_client_can_be_made(self):
        context = make_context(create_ros2_action_client=None)

        with self.assertRaises(NotImplementedError) as caught:
            context.create_ros2_action_client('sweep_the_field', FakeCodec())

        assert 'sweep_the_field' in str(caught.exception)


if __name__ == '__main__':
    unittest.main()
