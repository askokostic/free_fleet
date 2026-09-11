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
What the adapter reports to RMF about a running action, and when it stops one.

An action commands the robot for as long as it runs, so it is the only thing
that knows which goal is driving the robot. Dropping it leaves that goal
running with nobody watching it, which is why every path that takes an action
away tells it first.

Nav2RobotAdapter.__init__ waits for a live robot to localize, so these build
the adapter without it and populate only the fields these paths read.
"""

import unittest

from free_fleet_adapter.action import RobotActionState
from free_fleet_adapter.nav2_robot_adapter import Nav2RobotAdapter
from free_fleet_adapter.robot_adapter import ExecutionHandle

from tests.fake_ros2_action_endpoint import FakeNode


class FakeExecution:
    """The RMF execution an action reports to, and which RMF can withdraw."""

    def __init__(self, okay=True):
        self._okay = okay
        self.finish_count = 0
        self.identifier = object()

    def okay(self) -> bool:
        return self._okay

    def finished(self):
        self.finish_count += 1

    def rmf_takes_it_away(self):
        self._okay = False


class FakeAction:
    """An action whose reported state a test sets, and which counts stops."""

    def __init__(self, state=RobotActionState.IN_PROGRESS):
        self.state = state
        self.updates = 0
        self.stops = 0

    def update_action(self) -> RobotActionState:
        self.updates += 1
        return self.state

    def stop_action(self):
        self.stops += 1


class FakeActionFactory:

    def __init__(self, action):
        self.action = action
        self.actions = ['some_action']

    def supports_action(self, category: str) -> bool:
        return True

    def perform_action(self, category, description, execution):
        return self.action


def make_adapter():
    adapter = Nav2RobotAdapter.__new__(Nav2RobotAdapter)
    adapter.name = 'robogreen'
    adapter.node = FakeNode()
    adapter.exec_handle = None
    adapter.nav_command = None
    adapter.action_to_plugin_name = {}
    adapter.action_factories = {}
    return adapter


def make_action(adapter, state=RobotActionState.IN_PROGRESS, okay=True):
    """Give the adapter a running action, and return it with its execution."""
    execution = FakeExecution(okay=okay)
    action = FakeAction(state)
    exec_handle = ExecutionHandle(execution)
    exec_handle.set_action(action)
    adapter.exec_handle = exec_handle
    return action, execution


class TestReportingAnActionsOutcome(unittest.TestCase):

    def test_a_running_action_keeps_the_robot(self):
        adapter = make_adapter()
        action, execution = make_action(adapter)

        adapter._report_action_outcome(adapter.exec_handle)

        assert action.updates == 1
        assert execution.finish_count == 0
        assert adapter.exec_handle is not None

    def test_an_action_winding_down_keeps_the_robot(self):
        adapter = make_adapter()
        action, execution = make_action(adapter, RobotActionState.CANCELING)

        adapter._report_action_outcome(adapter.exec_handle)

        assert execution.finish_count == 0
        assert adapter.exec_handle is not None

    def test_a_completed_action_finishes_its_execution(self):
        adapter = make_adapter()
        action, execution = make_action(adapter, RobotActionState.COMPLETED)

        adapter._report_action_outcome(adapter.exec_handle)

        assert execution.finish_count == 1
        assert adapter.exec_handle is None

    def test_a_failed_action_finishes_its_execution(self):
        adapter = make_adapter()
        action, execution = make_action(adapter, RobotActionState.FAILED)

        adapter._report_action_outcome(adapter.exec_handle)

        assert execution.finish_count == 1
        assert adapter.exec_handle is None

    def test_a_cancelled_action_is_never_reported_as_finished(self):
        # RMF cancels the execution itself and gives the event its terminal
        # status; finishing it here would stamp Completed over that
        adapter = make_adapter()
        action, execution = make_action(adapter, RobotActionState.CANCELED)

        adapter._report_action_outcome(adapter.exec_handle)

        assert execution.finish_count == 0
        assert adapter.exec_handle is None


class TestTellingAnActionRmfHasGivenUpOnIt(unittest.TestCase):

    def test_an_execution_rmf_withdrew_stops_the_action(self):
        adapter = make_adapter()
        action, execution = make_action(adapter)
        execution.rmf_takes_it_away()

        adapter._report_action_outcome(adapter.exec_handle)

        assert action.stops == 1

    def test_the_action_is_told_once_however_often_it_is_updated(self):
        adapter = make_adapter()
        action, execution = make_action(adapter)
        execution.rmf_takes_it_away()

        adapter._report_action_outcome(adapter.exec_handle)
        adapter._report_action_outcome(adapter.exec_handle)
        adapter._report_action_outcome(adapter.exec_handle)

        assert action.stops == 1

    def test_a_withdrawn_execution_is_still_updated_until_it_stops(self):
        # The robot's goal is still running, so the action keeps command of it
        adapter = make_adapter()
        action, execution = make_action(adapter)
        execution.rmf_takes_it_away()

        adapter._report_action_outcome(adapter.exec_handle)

        assert action.updates == 1
        assert adapter.exec_handle is not None

    def test_an_action_rmf_still_wants_is_not_stopped(self):
        adapter = make_adapter()
        action, execution = make_action(adapter)

        adapter._report_action_outcome(adapter.exec_handle)

        assert action.stops == 0


class TestStoppingAnActivity(unittest.TestCase):

    def test_a_stop_reaches_the_running_action(self):
        adapter = make_adapter()
        action, execution = make_action(adapter)

        adapter.stop(adapter.exec_handle.activity)

        assert action.stops == 1

    def test_a_stopped_action_keeps_the_robot_until_it_says_it_stopped(self):
        # Dropping it here would leave its goal running, unwatched
        adapter = make_adapter()
        action, execution = make_action(adapter)

        adapter.stop(adapter.exec_handle.activity)

        assert adapter.exec_handle is not None
        assert execution.finish_count == 0

    def test_a_stopped_action_is_told_once(self):
        adapter = make_adapter()
        action, execution = make_action(adapter)

        adapter.stop(adapter.exec_handle.activity)
        adapter.stop(adapter.exec_handle.activity)

        assert action.stops == 1


class TestStoppingWhenThereIsNoActionToStop(unittest.TestCase):

    def test_a_handle_with_no_action_does_not_swallow_the_navigation_stop(
            self):
        # execute_action clears the action but leaves the handle behind if it
        # then refuses the incoming category
        adapter = make_adapter()
        action, execution = make_action(adapter)
        adapter.exec_handle.action = None
        stopped = []
        adapter.nav_command = type(
            'NavCommand', (), {
                'exec_handle': ExecutionHandle(FakeExecution()),
                'client': type(
                    'Client', (), {'cancel': lambda self: stopped.append(1)})(),
            })()
        adapter.nav_command.exec_handle.set_goal_id(None)

        adapter.stop(adapter.exec_handle.activity)

        assert stopped == [1]


class TestANewActionArrivingWhileOneRuns(unittest.TestCase):

    def test_the_outgoing_action_is_told_to_stop(self):
        adapter = make_adapter()
        outgoing, execution = make_action(adapter)
        incoming = FakeAction()
        adapter.action_factories = {'plugin': FakeActionFactory(incoming)}

        adapter.execute_action('some_action', {}, FakeExecution())

        assert outgoing.stops == 1
        assert adapter.exec_handle.action is incoming


if __name__ == '__main__':
    unittest.main()
