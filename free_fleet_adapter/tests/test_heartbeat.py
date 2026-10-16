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

"""The heartbeat to the robot: on only with heartbeat_topic."""

import unittest
from unittest import mock

from rclpy.serialization import serialize_message
from std_msgs.msg import Empty

from tests.adapter_doubles import make_adapter


def start_heartbeat(robot_config):
    adapter = make_adapter()
    adapter.robot_config_yaml = robot_config
    adapter.zenoh_session = mock.MagicMock()
    adapter.node = mock.MagicMock()
    adapter._start_heartbeat()
    return adapter


class TestTheHeartbeat(unittest.TestCase):

    def test_without_a_topic_there_is_no_heartbeat(self):
        adapter = start_heartbeat({})
        adapter.zenoh_session.declare_publisher.assert_not_called()
        adapter.node.create_timer.assert_not_called()

    def test_the_heartbeat_is_an_empty_message_on_the_robot_topic(self):
        adapter = start_heartbeat({
            'heartbeat_topic': 'zone_manager/heartbeat',
            'heartbeat_period_sec': 0.5,
        })
        adapter.zenoh_session.declare_publisher.assert_called_once_with(
            'robogreen/zone_manager/heartbeat')
        period, callback = adapter.node.create_timer.call_args.args
        assert period == 0.5

        callback()

        publisher = adapter.zenoh_session.declare_publisher.return_value
        publisher.put.assert_called_once_with(serialize_message(Empty()))

    def test_the_default_period_is_one_second(self):
        adapter = start_heartbeat({'heartbeat_topic': 'heartbeat'})
        period, _ = adapter.node.create_timer.call_args.args
        assert period == 1.0


if __name__ == '__main__':
    unittest.main()
