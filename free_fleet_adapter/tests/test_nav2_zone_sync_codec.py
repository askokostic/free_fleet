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
The hand-written zone sync types, checked against the real services.

The adapter reads CDR over zenoh and does not load nav2_zone_manager. Thus
these tests give a message that the generated type writes to the hand-written
type, and the other way around. Only the tests import nav2_zone_manager.
Without nav2_zone_manager or rmf_zone_manager_msgs, the tests are skipped.
"""

import unittest

import pytest

pytest.importorskip('rmf_zone_manager_msgs')
pytest.importorskip('nav2_zone_manager')

from free_fleet_adapter.nav2_zone_sync import (  # noqa: E402, I100
    GetZoneStates_Request,
    GetZoneStates_Response,
    SetZoneState_Request,
    SetZoneState_Response,
)
from geometry_msgs.msg import Point32  # noqa: E402
from nav2_zone_manager.msg import ZoneState  # noqa: E402
from nav2_zone_manager.srv import GetZoneStates, SetZoneState  # noqa: E402
from rclpy.serialization import (  # noqa: E402
    deserialize_message,
    serialize_message,
)


class TestTheSetZoneStateCall(unittest.TestCase):

    def test_the_request_is_read_by_the_generated_type(self):
        for enabled in (True, False):
            request = deserialize_message(
                SetZoneState_Request('zone_a', enabled).serialize(),
                SetZoneState.Request)
            assert request.name == 'zone_a'
            assert request.enabled is enabled

    def test_the_response_of_the_generated_type_is_read(self):
        response = SetZoneState_Response.deserialize(serialize_message(
            SetZoneState.Response(success=True, message='Enabled')))
        assert response.success is True
        assert response.message == 'Enabled'


class TestTheGetZoneStatesCall(unittest.TestCase):

    def test_the_request_is_read_by_the_generated_type(self):
        deserialize_message(
            GetZoneStates_Request().serialize(), GetZoneStates.Request)
        assert GetZoneStates_Request().serialize() == \
            serialize_message(GetZoneStates.Request())

    def test_the_response_of_the_generated_type_is_read(self):
        message = GetZoneStates.Response()
        message.zone_states.header.stamp.sec = 12
        message.zone_states.header.frame_id = 'map'
        for name, enabled in (('zone_a', True), ('zone_b', False)):
            zone = ZoneState(
                name=name, enabled=enabled, zone_type='keepout',
                priority=100, description='a zone',
                evacuate_on_link_loss=enabled)
            zone.polygon.points = [
                Point32(x=1.5, y=2.0, z=0.0), Point32(x=3.0, y=4.0, z=0.0),
                Point32(x=5.0, y=6.0, z=0.0)]
            message.zone_states.zones.append(zone)

        response = GetZoneStates_Response.deserialize(
            serialize_message(message))

        states = response.zone_states
        assert states.header.stamp.sec == 12
        assert states.header.frame_id == 'map'
        assert [(z.name, z.enabled) for z in states.zones] == \
            [('zone_a', True), ('zone_b', False)]
        assert states.zones[0].zone_type == 'keepout'
        assert states.zones[0].priority == 100
        assert states.zones[0].description == 'a zone'
        assert states.zones[1].polygon.points[2].y == 6.0
        assert states.zones[0].evacuate_on_link_loss
        assert not states.zones[1].evacuate_on_link_loss


if __name__ == '__main__':
    unittest.main()
