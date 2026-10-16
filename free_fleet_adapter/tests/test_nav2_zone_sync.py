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
The zone sync: the robot zones follow the RMF zones closed with LEAVE_NOW.

The sync turns off only the zones that it turned on or adopted. A zone that
another user turned on stays on, and RMF shows an issue. Without
rmf_zone_manager_msgs, the tests are skipped: the adapter needs it only with
nav2_zone_sync.
"""

import unittest
from unittest import mock

import pytest

pytest.importorskip('rmf_zone_manager_msgs')

from free_fleet.ros2_types import Header, Time  # noqa: E402, I100
from free_fleet_adapter.nav2_zone_sync import (  # noqa: E402
    GeometryMsgs_Polygon,
    GetZoneStates_Response,
    Nav2ZoneSync,
    SetZoneState_Request,
    SetZoneState_Response,
    zone_sync_actions,
    ZoneState,
    ZoneStates,
)
from rmf_zone_manager_msgs.msg import FleetZoneState  # noqa: E402
from rmf_zone_manager_msgs.msg import ZoneState as RmfZoneState  # noqa: E402
from rmf_zone_manager_msgs.msg import ZoneStates as RmfZoneStates  # noqa: E402

from tests.adapter_doubles import FakeUpdateHandle  # noqa: E402
from tests.fake_ros2_action_endpoint import FakeNode, FakeReply  # noqa: E402

LEAVE_NOW = FleetZoneState.EVACUATION_LEAVE_NOW
NONE = FleetZoneState.EVACUATION_NONE


class FakeRobotZoneManager:
    """The nav2_zone_manager of one robot, reached over a zenoh session."""

    def __init__(self, zones=('zone_a', 'zone_b')):
        self.enabled = {zone: False for zone in zones}
        self.answers = True
        self.refuse_sets = False
        self.sets = []

    def get(self, key_expr, *, payload=None, timeout=None):
        if not self.answers:
            return []
        if key_expr == 'robogreen/get_zone_states':
            zones = [
                ZoneState(name, enabled, 'keepout', 100, '',
                          GeometryMsgs_Polygon([]), False)
                for name, enabled in self.enabled.items()]
            response = GetZoneStates_Response(
                ZoneStates(Header(Time(0, 0), 'map'), zones))
            return [FakeReply(ok=response.serialize())]
        assert key_expr == 'robogreen/set_zone_state'
        request = SetZoneState_Request.deserialize(payload)
        self.sets.append((request.name, request.enabled))
        success = not self.refuse_sets and request.name in self.enabled
        if success:
            self.enabled[request.name] = request.enabled
        return [FakeReply(
            ok=SetZoneState_Response(success, 'scripted').serialize())]

    def on(self):
        return {name for name, enabled in self.enabled.items() if enabled}


def make_sync(robot):
    sync = Nav2ZoneSync(robot, FakeNode(), 'robogreen')
    sync.add_robot('robogreen', FakeUpdateHandle())
    return sync


def rmf_closes(sync, **evacuation_of_zone):
    """Let zone_states arrive with these zones closed for the fleet."""
    msg = RmfZoneStates()
    for name, evacuation in evacuation_of_zone.items():
        fleet = FleetZoneState(
            fleet_name='robogreen', closed=True, evacuation=evacuation)
        other_fleet = FleetZoneState(
            fleet_name='other', closed=True, evacuation=LEAVE_NOW)
        msg.zones.append(RmfZoneState(name=name, fleets=[fleet, other_fleet]))
    sync.zone_states_callback(msg)


def issues(sync):
    handle = sync.robots[0].update_handle
    return [category for _, category, _ in handle.more().issues]


class TestTheRule(unittest.TestCase):

    def test_each_set_of_zones(self):
        actions = zone_sync_actions(
            required={'a', 'b'}, on={'b', 'c', 'd'}, recorded={'c', 'e'})
        assert actions.enable == {'a'}
        assert actions.adopt == {'b'}
        assert actions.disable == {'c'}
        assert actions.forget == {'e'}
        assert actions.foreign == {'d'}


class TestTheZoneSync(unittest.TestCase):

    def test_nothing_is_set_before_the_first_zone_states(self):
        robot = FakeRobotZoneManager()
        robot.enabled['zone_b'] = True
        sync = make_sync(robot)
        sync.sync()
        assert robot.sets == []
        assert issues(sync) == []

    def test_leave_now_turns_the_zone_on_and_none_turns_it_off(self):
        robot = FakeRobotZoneManager()
        sync = make_sync(robot)
        rmf_closes(sync, zone_a=LEAVE_NOW)
        sync.sync()
        assert robot.on() == {'zone_a'}
        assert sync.robots[0].recorded == {'zone_a'}

        rmf_closes(sync, zone_a=NONE)
        sync.sync()
        assert robot.on() == set()
        assert sync.robots[0].recorded == set()

    def test_open_turns_the_zone_off(self):
        robot = FakeRobotZoneManager()
        sync = make_sync(robot)
        rmf_closes(sync, zone_a=LEAVE_NOW)
        sync.sync()
        rmf_closes(sync)
        sync.sync()
        assert robot.on() == set()

    def test_a_local_disable_is_turned_on_again(self):
        robot = FakeRobotZoneManager()
        sync = make_sync(robot)
        rmf_closes(sync, zone_a=LEAVE_NOW)
        sync.sync()
        robot.enabled['zone_a'] = False
        sync.sync()
        assert robot.on() == {'zone_a'}

    def test_after_a_robot_reboot_the_zone_is_turned_on_again(self):
        robot = FakeRobotZoneManager()
        sync = make_sync(robot)
        rmf_closes(sync, zone_a=LEAVE_NOW)
        sync.sync()
        robot.enabled = {'zone_a': False, 'zone_b': False}
        sync.sync()
        assert robot.on() == {'zone_a'}

    def test_after_an_adapter_restart_a_required_zone_is_adopted(self):
        robot = FakeRobotZoneManager()
        robot.enabled['zone_a'] = True
        sync = make_sync(robot)
        rmf_closes(sync, zone_a=LEAVE_NOW)
        sync.sync()
        assert robot.sets == []
        assert sync.robots[0].recorded == {'zone_a'}

        rmf_closes(sync)
        sync.sync()
        assert robot.on() == set()

    def test_a_foreign_zone_stays_on_with_an_issue(self):
        robot = FakeRobotZoneManager()
        robot.enabled['zone_b'] = True
        sync = make_sync(robot)
        rmf_closes(sync)
        sync.sync()
        sync.sync()
        assert robot.on() == {'zone_b'}
        assert robot.sets == []
        assert issues(sync) == ['foreign_robot_zones']

        robot.enabled['zone_b'] = False
        sync.sync()
        assert sync.robots[0].foreign_issue is None

    def test_a_zone_off_on_the_robot_is_forgotten(self):
        robot = FakeRobotZoneManager()
        sync = make_sync(robot)
        rmf_closes(sync, zone_a=LEAVE_NOW)
        sync.sync()
        robot.enabled['zone_a'] = False
        rmf_closes(sync)
        sync.sync()
        assert robot.sets == [('zone_a', True)]
        assert sync.robots[0].recorded == set()

    def test_a_lost_reply_changes_nothing(self):
        robot = FakeRobotZoneManager()
        sync = make_sync(robot)
        rmf_closes(sync, zone_a=LEAVE_NOW)
        sync.sync()
        robot.answers = False
        rmf_closes(sync)
        sync.sync()
        assert sync.robots[0].recorded == {'zone_a'}
        assert sync.node.get_logger().has('warn', 'no zone states')

        robot.answers = True
        sync.sync()
        assert robot.on() == set()
        assert sync.robots[0].recorded == set()

    def test_no_answer_with_a_required_zone_raises_an_issue(self):
        robot = FakeRobotZoneManager()
        robot.answers = False
        sync = make_sync(robot)
        sync.sync()
        assert issues(sync) == []

        rmf_closes(sync, zone_a=LEAVE_NOW)
        sync.sync()
        sync.sync()
        assert issues(sync) == ['evacuation_not_applied']
        assert 'does not answer' in sync.robots[0].failed_issue_message

        robot.answers = True
        sync.sync()
        assert robot.on() == {'zone_a'}
        assert sync.robots[0].failed_issue is None
        assert issues(sync) == ['evacuation_not_applied']

    def test_no_answer_without_a_required_zone_raises_no_issue(self):
        robot = FakeRobotZoneManager()
        robot.answers = False
        sync = make_sync(robot)
        rmf_closes(sync, zone_a=NONE)
        sync.sync()
        assert issues(sync) == []

    def test_an_exception_does_not_stop_the_sync_thread(self):
        sync = make_sync(FakeRobotZoneManager())
        sync.period_sec = 0.0
        calls = []

        def failing_sync():
            calls.append(None)
            raise RuntimeError('scripted')

        sync.sync = failing_sync
        with mock.patch(
                'free_fleet_adapter.nav2_zone_sync.rclpy.ok',
                side_effect=[True, True, False]):
            sync._run()
        assert len(calls) == 2
        assert sync.node.get_logger().has('error', 'scripted')

    def test_a_failed_set_raises_an_issue_and_is_tried_again(self):
        robot = FakeRobotZoneManager()
        robot.refuse_sets = True
        sync = make_sync(robot)
        rmf_closes(sync, zone_a=LEAVE_NOW)
        sync.sync()
        sync.sync()
        assert robot.sets == [('zone_a', True), ('zone_a', True)]
        assert sync.robots[0].recorded == set()
        assert issues(sync) == ['evacuation_not_applied']

        robot.refuse_sets = False
        sync.sync()
        assert robot.on() == {'zone_a'}
        assert sync.robots[0].failed_issue is None

    def test_a_failed_disable_keeps_the_record(self):
        robot = FakeRobotZoneManager()
        sync = make_sync(robot)
        rmf_closes(sync, zone_a=LEAVE_NOW)
        sync.sync()
        robot.refuse_sets = True
        rmf_closes(sync)
        sync.sync()
        assert sync.robots[0].recorded == {'zone_a'}
        assert issues(sync) == ['evacuation_not_applied']

    def test_the_issue_follows_the_failed_zones(self):
        robot = FakeRobotZoneManager()
        sync = make_sync(robot)
        rmf_closes(sync, zone_c=LEAVE_NOW)
        sync.sync()
        rmf_closes(sync, zone_a=LEAVE_NOW, zone_b=LEAVE_NOW)
        first_issue = sync.robots[0].failed_issue
        robot.refuse_sets = True
        sync.sync()
        handle = sync.robots[0].update_handle
        assert [details['zones'] for _, _, details in handle.more().issues] \
            == [['zone_c'], ['zone_a', 'zone_b']]
        assert first_issue.resolved
        assert "['zone_a', 'zone_b']" in sync.robots[0].failed_issue_message

    def test_a_zone_unknown_to_the_robot_raises_an_issue(self):
        robot = FakeRobotZoneManager()
        sync = make_sync(robot)
        rmf_closes(sync, zone_c=LEAVE_NOW)
        sync.sync()
        assert issues(sync) == ['evacuation_not_applied']


if __name__ == '__main__':
    unittest.main()
