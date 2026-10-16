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
The gate: no new goal while the evacuation monitor moves the robot.

The evacuation monitor on the robot cancels all goals before its exit goal. A
goal of RMF or of a plugin would then replace or stop the exit goal. Thus the
adapter holds new goals while the monitor status shows an evacuation, a hold
or a failure, and while the status is too old.
"""

import unittest

from free_fleet.ros2_types import (
    GeometryMsgs_Point,
    GeometryMsgs_Pose,
    GeometryMsgs_PoseStamped,
    GeometryMsgs_Quaternion,
    Header,
    StdSrvs_SetBool_Request,
    StdSrvs_SetBool_Response,
    Time,
)
from free_fleet_adapter.action import RobotActionState
from free_fleet_adapter.nav2_robot_adapter import (
    ActionClaim,
    EvacuationStatus,
    Nav2RobotAdapter,
)
from free_fleet_adapter.ros2_action_client import (
    RequestRejected,
    Ros2GoalState,
)
from std_srvs.srv import SetBool

from tests.adapter_doubles import (
    FakeAction,
    FakeActionClient,
    FakeActionFactory,
    FakeDestination,
    FakeExecution,
    give_the_robot_a_navigation,
    give_the_robot_an_action,
    make_adapter as make_adapter_without_monitor,
    wind_the_clock_on,
)
from tests.fake_ros2_action_endpoint import (
    FakeCodec,
    FakeReply,
    send_goal_response,
)

HOLD = 'evacuation_monitor/hold'


def make_adapter(**kwargs):
    return make_adapter_without_monitor(use_evacuation_monitor=True, **kwargs)


def make_status(state='IDLE', inside_zone=False, goal_in_progress=False):
    pose = GeometryMsgs_PoseStamped(
        Header(Time(0, 0), 'map'),
        GeometryMsgs_Pose(
            GeometryMsgs_Point(0.0, 0.0, 0.0), GeometryMsgs_Quaternion()))
    return EvacuationStatus(
        header=Header(Time(0, 0), 'map'),
        state=state,
        inside_zone=inside_zone,
        evacuation_goal_in_progress=goal_in_progress,
        active_zone_name='zone_a' if inside_zone else '',
        goal_pose=pose,
        next_retry_time=Time(0, 0),
        detail='scripted',
    )


def robot_reports(adapter, **kwargs):
    """Let the evacuation monitor status arrive now."""
    status = make_status(**kwargs)
    adapter.evacuation_status = status
    adapter.evacuation_status_sec = adapter._monotonic_sec()
    if status.state != 'WAITING_FOR_DATA':
        adapter.evacuation_monitor_had_data = True


def a_navigation_arrives(adapter):
    execution = FakeExecution()
    adapter.navigate(FakeDestination(), execution)
    return execution


def monitor_answers(adapter, success=True):
    adapter.zenoh_session.replies[HOLD] = [FakeReply(
        ok=StdSrvs_SetBool_Response(success, 'scripted').serialize())]


def call_hold(adapter, hold):
    return adapter._hold_callback(SetBool.Request(data=hold), SetBool.Response())


def issue_categories(adapter):
    return [category for _, category, _ in adapter.update_handle.more().issues]


class TestTheGateConditions(unittest.TestCase):

    def test_no_status_yet_keeps_the_gate_open(self):
        assert make_adapter()._gate_reason() is None

    def test_a_robot_outside_the_zones_keeps_the_gate_open(self):
        adapter = make_adapter()
        robot_reports(adapter)
        assert adapter._gate_reason() is None

    def test_a_robot_in_a_zone_closes_the_gate(self):
        adapter = make_adapter()
        robot_reports(adapter, state='CANCELING', inside_zone=True)
        assert 'zone_a' in adapter._gate_reason()

    def test_an_exit_goal_in_progress_closes_the_gate(self):
        adapter = make_adapter()
        robot_reports(adapter, state='NAVIGATING', goal_in_progress=True)
        assert adapter._gate_reason() is not None

    def test_hold_and_failed_close_the_gate(self):
        for state in ('HOLD', 'FAILED'):
            adapter = make_adapter()
            robot_reports(adapter, state=state)
            assert state in adapter._gate_reason()

    def test_a_status_that_is_too_old_closes_the_gate(self):
        adapter = make_adapter()
        robot_reports(adapter)
        wind_the_clock_on(adapter, 3)
        assert adapter._gate_reason() is None
        wind_the_clock_on(adapter, 1)
        assert 'too old' in adapter._gate_reason()

    def test_a_monitor_without_pose_after_its_data_closes_the_gate(self):
        # Without a pose, the monitor reports inside_zone false
        adapter = make_adapter()
        robot_reports(adapter, state='WAITING_FOR_DATA')
        assert adapter._gate_reason() is None

        robot_reports(adapter, state='CANCELING', inside_zone=True)
        robot_reports(adapter, state='WAITING_FOR_DATA')
        assert 'no robot pose' in adapter._gate_reason()

        robot_reports(adapter)
        assert adapter._gate_reason() is None

    def test_a_remote_hold_closes_the_gate(self):
        adapter = make_adapter()
        adapter.remote_hold = True
        assert adapter._gate_reason() is not None


class TestTheGateHoldsTheGoals(unittest.TestCase):

    def test_a_navigation_waits_while_the_gate_is_closed(self):
        adapter = make_adapter()
        robot_reports(adapter, state='NAVIGATING', inside_zone=True)

        a_navigation_arrives(adapter)
        adapter.update(None)

        assert adapter.nav_client.started == []
        assert adapter.pending_command is not None

    def test_the_waiting_navigation_is_dropped_and_rmf_replans(self):
        # RMF planned it from where the robot was before the exit
        adapter = make_adapter()
        robot_reports(adapter, state='NAVIGATING', inside_zone=True)
        a_navigation_arrives(adapter)
        adapter.update(None)

        robot_reports(adapter)
        adapter.update(None)

        assert adapter.nav_client.started == []
        assert adapter.pending_command is None
        assert adapter.update_handle.more().replan_count == 1

    def test_each_dropped_navigation_is_planned_again(self):
        # Also more than MAX_REPLAN_COUNTS times: else RMF waits for ever
        adapter = make_adapter()
        for _ in range(Nav2RobotAdapter.MAX_REPLAN_COUNTS + 2):
            robot_reports(adapter, state='NAVIGATING', inside_zone=True)
            a_navigation_arrives(adapter)
            adapter.update(None)
            robot_reports(adapter)
            adapter.update(None)

        assert adapter.update_handle.more().replan_count == \
            Nav2RobotAdapter.MAX_REPLAN_COUNTS + 2

    def test_a_gate_that_opens_with_nothing_waiting_does_not_replan(self):
        # For example after a short link drop. A replan would also stop the
        # navigation goal that runs.
        adapter = make_adapter()
        robot_reports(adapter)
        adapter.update(None)
        wind_the_clock_on(adapter, 4)
        adapter.update(None)
        assert adapter.gate_closed

        robot_reports(adapter)
        adapter.update(None)

        assert adapter.update_handle.more().replan_count == 0

    def test_a_waiting_action_starts_when_the_gate_opens(self):
        # RMF does not plan an action again, thus it is not dropped
        adapter = make_adapter()
        action = FakeAction()
        adapter.action_factories = {'plugin': FakeActionFactory(action)}
        robot_reports(adapter, state='NAVIGATING', inside_zone=True)
        execution = FakeExecution()
        adapter.execute_action('some_action', {}, execution)
        adapter.update(None)
        assert adapter.claim is None
        assert adapter.pending_command is not None

        robot_reports(adapter)
        adapter.update(None)

        assert isinstance(adapter.claim, ActionClaim)
        assert adapter.claim.exec_handle.execution is execution
        assert adapter.update_handle.more().replan_count == 0

    def test_a_navigation_starts_at_once_while_the_gate_is_open(self):
        adapter = make_adapter()
        robot_reports(adapter)

        a_navigation_arrives(adapter)

        assert len(adapter.nav_client.started) == 1

    def test_a_plugin_goal_is_rejected_while_the_gate_is_closed(self):
        adapter = make_adapter()
        adapter.action_clients = {}
        adapter.zenoh_session.replies['send_goal'] = [
            FakeReply(ok=send_goal_response(True))]
        client = Nav2RobotAdapter.create_ros2_action_client(
            adapter, 'navigate_complete_coverage', FakeCodec())
        robot_reports(adapter, state='HOLD')

        with self.assertRaises(RequestRejected):
            client.start({'field': 'north'})
        assert not adapter.zenoh_session.queries_for('send_goal')

        robot_reports(adapter)
        assert client.start({'field': 'north'}) == client.goal_id


class TestAGoalCanceledOnTheRobot(unittest.TestCase):

    def test_the_monitor_cancel_is_not_an_arrival_and_rmf_replans(self):
        adapter = make_adapter()
        robot_reports(adapter)
        claim, client, execution = give_the_robot_a_navigation(
            adapter, Ros2GoalState.CANCELED)

        adapter.update(None)
        assert execution.finish_count == 0
        assert adapter.claim is None
        assert adapter._gate_reason() is not None

        robot_reports(adapter, state='NAVIGATING', inside_zone=True,
                      goal_in_progress=True)
        wind_the_clock_on(adapter, 1)
        adapter.update(None)
        a_navigation_arrives(adapter)
        wind_the_clock_on(adapter, 5)
        robot_reports(adapter, state='NAVIGATING', goal_in_progress=True)
        adapter.update(None)
        assert client.started == []

        robot_reports(adapter)
        adapter.update(None)
        assert adapter.pending_command is None
        assert adapter.update_handle.more().replan_count == 1
        assert adapter._gate_reason() is None
        assert issue_categories(adapter) == []

    def test_a_monitor_cancel_that_is_reported_after_the_exit_is_no_stop(self):
        # The client asks for the result only after the feedback ETA, thus
        # the cancel can be reported after the evacuation
        adapter = make_adapter()
        robot_reports(adapter)
        _, client, execution = give_the_robot_a_navigation(adapter)
        robot_reports(adapter, state='NAVIGATING', inside_zone=True,
                      goal_in_progress=True)
        adapter.update(None)
        robot_reports(adapter)
        adapter.update(None)
        assert adapter.update_handle.more().replan_count == 0

        client.reports(Ros2GoalState.CANCELED)
        adapter.update(None)
        wind_the_clock_on(adapter, 5)
        robot_reports(adapter)
        adapter.update(None)

        assert execution.finish_count == 0
        assert adapter.claim is None
        assert adapter._gate_reason() is None
        assert issue_categories(adapter) == []
        assert adapter.update_handle.more().replan_count == 1

    def test_after_the_window_without_an_evacuation_rmf_replans(self):
        # Another node canceled the goal, and the monitor did not evacuate
        adapter = make_adapter()
        robot_reports(adapter)
        give_the_robot_a_navigation(adapter, Ros2GoalState.CANCELED)
        adapter.update(None)
        assert adapter._gate_reason() is not None
        assert adapter.update_handle.more().replan_count == 0

        wind_the_clock_on(adapter, 2)
        robot_reports(adapter)
        adapter.update(None)

        assert adapter._gate_reason() is None
        assert adapter.update_handle.more().replan_count == 1
        assert issue_categories(adapter) == []

    def test_a_release_in_the_decision_window_opens_the_gate(self):
        adapter = make_adapter()
        robot_reports(adapter)
        give_the_robot_a_navigation(adapter, Ros2GoalState.CANCELED)
        adapter.update(None)

        wind_the_clock_on(adapter, 1)
        monitor_answers(adapter)
        call_hold(adapter, False)
        wind_the_clock_on(adapter, 1)
        robot_reports(adapter)
        adapter.update(None)

        assert adapter._gate_reason() is None
        assert issue_categories(adapter) == []

    def test_a_cancel_seen_after_a_new_command_replans_after_the_window(self):
        # The client can see the cancel late, after the feedback ETA. RMF
        # sends a new navigation first, and the endpoint answers the adapter
        # cancel that the goal was already over
        adapter = make_adapter()
        robot_reports(adapter)
        _, client, execution = give_the_robot_a_navigation(adapter)
        client.goal_was_over = True
        a_navigation_arrives(adapter)
        assert client.cancels == 1

        client.reports(Ros2GoalState.CANCELED)
        adapter.update(None)
        assert execution.finish_count == 0
        assert adapter.claim is None
        assert client.started == []

        wind_the_clock_on(adapter, 2)
        robot_reports(adapter)
        adapter.update(None)

        # RMF planned the waiting navigation before the cancel: it plans again
        assert adapter._gate_reason() is None
        assert adapter.pending_command is None
        assert adapter.update_handle.more().replan_count == 1
        assert client.started == []
        assert issue_categories(adapter) == []

    def test_a_failed_action_keeps_the_gate_open(self):
        # A plugin FAILED does not tell a cancel on the robot from other
        # failures (issue 10)
        adapter = make_adapter()
        robot_reports(adapter)
        _, execution = give_the_robot_an_action(
            adapter, RobotActionState.FAILED, okay=False)
        adapter.update(None)
        assert execution.finish_count == 1

        wind_the_clock_on(adapter, 2)
        robot_reports(adapter)
        adapter.update(None)
        assert issue_categories(adapter) == []
        assert adapter._gate_reason() is None

    def test_a_canceled_action_closes_the_gate_for_the_window(self):
        # The adapter did not ask the action to stop (issue 10)
        adapter = make_adapter()
        robot_reports(adapter)
        _, execution = give_the_robot_an_action(
            adapter, RobotActionState.CANCELED, okay=True)
        adapter.update(None)
        assert execution.finish_count == 0
        assert adapter.claim is None
        assert adapter._gate_reason() is not None

        wind_the_clock_on(adapter, 2)
        robot_reports(adapter)
        adapter.update(None)

        assert adapter._gate_reason() is None
        assert issue_categories(adapter) == []

    def test_an_action_that_the_monitor_stopped_keeps_the_gate_open(self):
        adapter = make_adapter()
        robot_reports(adapter)
        action, _ = give_the_robot_an_action(adapter)
        robot_reports(adapter, state='NAVIGATING', inside_zone=True,
                      goal_in_progress=True)
        adapter.update(None)

        action.state = RobotActionState.FAILED
        robot_reports(adapter)
        adapter.update(None)
        wind_the_clock_on(adapter, 2)
        robot_reports(adapter)
        adapter.update(None)

        assert issue_categories(adapter) == []
        assert adapter._gate_reason() is None

    def test_an_action_that_rmf_canceled_keeps_the_gate_open(self):
        adapter = make_adapter()
        robot_reports(adapter)
        _, execution = give_the_robot_an_action(
            adapter, RobotActionState.CANCELED, okay=False)
        adapter.update(None)
        assert execution.finish_count == 0

        wind_the_clock_on(adapter, 2)
        robot_reports(adapter)
        adapter.update(None)

        assert issue_categories(adapter) == []
        assert adapter._gate_reason() is None

    def test_a_stop_that_is_not_reported_raises_an_issue_at_the_gate(self):
        adapter = make_adapter(stop_timeout_sec=5.0)
        robot_reports(adapter)
        give_the_robot_an_action(adapter)
        a_navigation_arrives(adapter)
        call_hold(adapter, True)

        wind_the_clock_on(adapter, 6)
        robot_reports(adapter)
        adapter.update(None)

        assert issue_categories(adapter) == ['robot_stop_timeout']
        (_, _, details), = adapter.update_handle.more().issues
        assert 'nothing is waiting' not in details['message']
        assert adapter.nav_client.started == []


class TestTheRemoteHold(unittest.TestCase):

    def test_a_hold_closes_the_gate_and_is_sent_to_the_monitor(self):
        adapter = make_adapter()
        monitor_answers(adapter)

        response = call_hold(adapter, True)

        assert response.success
        assert adapter._gate_reason() is not None
        (_, payload), = adapter.zenoh_session.queries
        assert StdSrvs_SetBool_Request.deserialize(payload).data is True

    def test_a_hold_closes_the_gate_also_without_the_monitor(self):
        adapter = make_adapter()

        response = call_hold(adapter, True)

        assert not response.success
        assert adapter._gate_reason() is not None

    def test_a_hold_without_the_monitor_stops_the_running_goal(self):
        adapter = make_adapter()
        robot_reports(adapter)
        _, client, execution = give_the_robot_a_navigation(adapter)

        call_hold(adapter, True)
        assert client.cancels == 1
        client.reports(Ros2GoalState.CANCELED)
        adapter.update(None)
        assert adapter.claim is None
        assert execution.finish_count == 0
        assert issue_categories(adapter) == []

        call_hold(adapter, False)
        robot_reports(adapter)
        adapter.update(None)
        assert adapter.update_handle.more().replan_count == 1

    def test_a_hold_fails_the_running_action(self):
        # A stopped action ends CANCELED, and RMF then waits for ever. Thus
        # the hold cancels the goals of the action, and the action fails.
        adapter = make_adapter()
        robot_reports(adapter)
        client = FakeActionClient(Ros2GoalState.EXECUTING)
        adapter.action_clients = {'navigate_complete_coverage': client}
        action, execution = give_the_robot_an_action(adapter)

        call_hold(adapter, True)
        assert client.cancels == 1
        assert action.stops == 0

        action.state = RobotActionState.FAILED
        robot_reports(adapter)
        adapter.update(None)
        assert adapter.claim is None
        assert execution.finish_count == 1
        assert adapter.cancel_window_end_sec is None

    def test_a_release_opens_the_gate(self):
        adapter = make_adapter()
        robot_reports(adapter)
        monitor_answers(adapter)
        call_hold(adapter, True)
        adapter.update(None)

        response = call_hold(adapter, False)
        robot_reports(adapter)
        adapter.update(None)

        assert response.success
        assert adapter._gate_reason() is None
        assert adapter.update_handle.more().replan_count == 0

    def test_after_a_hold_during_a_navigation_rmf_replans(self):
        adapter = make_adapter()
        robot_reports(adapter)
        _, client, execution = give_the_robot_a_navigation(adapter)
        monitor_answers(adapter)
        call_hold(adapter, True)
        client.reports(Ros2GoalState.CANCELED)
        robot_reports(adapter, state='HOLD')
        adapter.update(None)
        assert execution.finish_count == 0

        monitor_answers(adapter)
        call_hold(adapter, False)
        robot_reports(adapter)
        adapter.update(None)

        assert adapter._gate_reason() is None
        assert adapter.update_handle.more().replan_count == 1
        assert issue_categories(adapter) == ['evacuation_monitor']

    def test_a_release_without_the_monitor_opens_the_gate(self):
        adapter = make_adapter()
        call_hold(adapter, True)

        response = call_hold(adapter, False)

        assert not response.success
        assert adapter._gate_reason() is None

    def test_a_monitor_that_stays_in_hold_keeps_the_gate_closed(self):
        adapter = make_adapter()
        monitor_answers(adapter)
        call_hold(adapter, True)
        adapter.zenoh_session.replies.clear()

        call_hold(adapter, False)
        robot_reports(adapter, state='HOLD')

        assert adapter._gate_reason() is not None


class TestTheIssues(unittest.TestCase):

    def test_no_status_raises_one_issue_until_a_status_arrives(self):
        adapter = make_adapter()
        adapter.update(None)
        assert issue_categories(adapter) == []

        wind_the_clock_on(adapter, 11)
        adapter.update(None)
        adapter.update(None)
        assert issue_categories(adapter) == [
            'evacuation_monitor_status_missing']

        ticket = adapter.no_status_issue
        robot_reports(adapter)
        adapter.update(None)
        assert ticket.resolved

    def test_a_status_that_stops_raises_one_issue_until_a_fresh_one(self):
        adapter = make_adapter()
        robot_reports(adapter)
        adapter.update(None)

        wind_the_clock_on(adapter, 10)
        adapter.update(None)
        assert issue_categories(adapter) == []
        wind_the_clock_on(adapter, 1)
        adapter.update(None)
        adapter.update(None)
        assert issue_categories(adapter) == [
            'evacuation_monitor_status_missing']

        ticket = adapter.no_status_issue
        robot_reports(adapter)
        adapter.update(None)
        assert ticket.resolved
        assert adapter.no_status_issue is None

    def test_failed_and_hold_each_raise_one_issue_until_they_end(self):
        adapter = make_adapter()
        robot_reports(adapter, state='FAILED', inside_zone=True)
        adapter.update(None)
        adapter.update(None)
        assert issue_categories(adapter) == ['evacuation_monitor']
        failed = adapter.monitor_issue

        robot_reports(adapter, state='HOLD')
        adapter.update(None)
        assert failed.resolved
        assert len(issue_categories(adapter)) == 2

        hold = adapter.monitor_issue
        robot_reports(adapter)
        adapter.update(None)
        assert hold.resolved
        assert adapter.monitor_issue is None


if __name__ == '__main__':
    unittest.main()


class TestWithoutTheEvacuationMonitor(unittest.TestCase):
    """use_evacuation_monitor is false: the adapter behaves as without a gate."""

    def test_no_issue_and_no_gate_without_a_status(self):
        adapter = make_adapter_without_monitor()
        wind_the_clock_on(adapter, 60)
        adapter.update(None)
        assert issue_categories(adapter) == []
        assert adapter._gate_reason() is None

    def test_a_status_does_not_close_the_gate(self):
        adapter = make_adapter_without_monitor()
        robot_reports(adapter, state='HOLD', inside_zone=True)
        assert adapter._gate_reason() is None

    def test_a_canceled_goal_replans_at_once(self):
        adapter = make_adapter_without_monitor()
        _, _, execution = give_the_robot_a_navigation(
            adapter, Ros2GoalState.CANCELED)
        adapter.update(None)
        assert execution.finish_count == 0
        assert adapter.update_handle.more().replan_count == 1
        assert adapter._gate_reason() is None
