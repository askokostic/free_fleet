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

"""
The hand-written evacuation types, checked against the real messages.

The adapter reads CDR over zenoh and does not load the message packages. Thus
these tests give a message that the generated type writes to the hand-written
type, and the other way around. Only the tests import the message packages.
Without nav2_evacuation_monitor, the tests are skipped: the adapter does not
need it.
"""

import unittest

from free_fleet.ros2_types import (
    StdSrvs_SetBool_Request,
    StdSrvs_SetBool_Response,
)
from free_fleet_adapter.nav2_robot_adapter import EvacuationStatus
import pytest
from rclpy.serialization import deserialize_message, serialize_message
from std_srvs.srv import SetBool

RosEvacuationStatus = pytest.importorskip(
    'nav2_evacuation_monitor.msg').EvacuationStatus


class TestTheEvacuationStatus(unittest.TestCase):

    def test_a_status_is_read_field_for_field(self):
        message = RosEvacuationStatus()
        message.header.stamp.sec = 12
        message.header.frame_id = 'map'
        message.state = 'NAVIGATING'
        message.inside_zone = True
        message.evacuation_goal_in_progress = True
        message.active_zone_name = 'zone_a'
        message.goal_pose.pose.position.x = 101.5
        message.next_retry_time.sec = 7
        message.detail = 'Evacuation NavigateToPose goal active'

        status = EvacuationStatus.deserialize(serialize_message(message))

        assert status.header.stamp.sec == 12
        assert status.header.frame_id == 'map'
        assert status.state == 'NAVIGATING'
        assert status.inside_zone is True
        assert status.evacuation_goal_in_progress is True
        assert status.active_zone_name == 'zone_a'
        assert status.goal_pose.pose.position.x == 101.5
        assert status.next_retry_time.sec == 7
        assert status.detail == 'Evacuation NavigateToPose goal active'


class TestTheHoldCall(unittest.TestCase):

    def test_the_request_is_read_by_the_generated_type(self):
        for data in (True, False):
            request = deserialize_message(
                StdSrvs_SetBool_Request(data=data).serialize(),
                SetBool.Request)
            assert request.data is data

    def test_the_response_of_the_generated_type_is_read(self):
        response = SetBool.Response(success=True, message='Hold')

        decoded = StdSrvs_SetBool_Response.deserialize(
            serialize_message(response))

        assert decoded.success is True
        assert decoded.message == 'Hold'


if __name__ == '__main__':
    unittest.main()
