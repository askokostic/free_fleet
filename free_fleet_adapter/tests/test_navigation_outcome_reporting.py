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
What the adapter reports to RMF about a navigation goal.

Nav2RobotAdapter.__init__ waits for a live robot to localize, so these build
the adapter without it and populate only the fields the reporting path reads.
"""

import unittest

from free_fleet_adapter.nav2_robot_adapter import (
    Nav2RobotAdapter,
    NavigationCommand,
)
from free_fleet_adapter.robot_adapter import ExecutionHandle
from free_fleet_adapter.ros2_action_client import (
    Ros2GoalOutcome,
    Ros2GoalState,
)

from tests.fake_ros2_action_endpoint import FakeNode

GOAL_ID = list(range(16))
OTHER_GOAL_ID = list(range(16, 32))


class FakeExecution:

    def __init__(self):
        self.finish_count = 0

    def finished(self):
        self.finish_count += 1


class FakeIssueTicket:

    def __init__(self):
        self.resolved = False

    def resolve(self, msg):
        self.resolved = True


class FakeUpdateHandleMore:

    def __init__(self):
        self.issues = []
        self.replan_count = 0

    def create_issue(self, tier, category, detail):
        self.issues.append((tier, category, detail))
        return FakeIssueTicket()

    def replan(self):
        self.replan_count += 1


class FakeUpdateHandle:

    def __init__(self):
        self._more = FakeUpdateHandleMore()

    def more(self):
        return self._more


class FakeActionClient:
    """A client that reports a scripted outcome for a scripted goal."""

    def __init__(self, state, goal_id=GOAL_ID, detail='scripted'):
        self.scripted = Ros2GoalOutcome(state, goal_id, detail)

    def outcome(self):
        return self.scripted


def make_adapter():
    adapter = Nav2RobotAdapter.__new__(Nav2RobotAdapter)
    adapter.name = 'robogreen'
    adapter.node = FakeNode()
    adapter.update_handle = FakeUpdateHandle()
    adapter.nav_command = None
    adapter.nav_issue_ticket = None
    adapter.replan_counts = 0
    return adapter


def make_command(adapter, state, client_goal_id=GOAL_ID,
                 command_goal_id=GOAL_ID):
    """Return the command and, separately, the execution it reports to."""
    execution = FakeExecution()
    exec_handle = ExecutionHandle(execution)
    exec_handle.set_goal_id(command_goal_id)
    command = NavigationCommand(
        FakeActionClient(state, client_goal_id), exec_handle)
    adapter.nav_command = command
    return command, execution


class TestReportingOnlyThisCommandsGoal(unittest.TestCase):

    def test_a_goal_that_has_not_been_sent_yet_is_not_reported(self):
        adapter = make_adapter()
        command, execution = make_command(
            adapter, Ros2GoalState.SUCCEEDED, command_goal_id=None)

        adapter._report_navigation_outcome(command)

        assert execution.finish_count == 0
        assert adapter.nav_command is command

    def test_another_goals_outcome_is_not_attributed_to_this_command(self):
        # The client is shared per endpoint, so it may still be reporting the
        # goal that ran before this command was given one
        adapter = make_adapter()
        command, execution = make_command(
            adapter,
            Ros2GoalState.SUCCEEDED,
            client_goal_id=OTHER_GOAL_ID,
            command_goal_id=GOAL_ID
        )

        adapter._report_navigation_outcome(command)

        assert execution.finish_count == 0
        assert adapter.nav_command is command

    def test_a_client_with_no_goal_does_not_complete_the_command(self):
        adapter = make_adapter()
        command, execution = make_command(adapter, Ros2GoalState.IDLE)

        adapter._report_navigation_outcome(command)

        assert execution.finish_count == 0


class TestReportingAnOutcome(unittest.TestCase):

    def test_a_running_goal_leaves_the_command_alone(self):
        adapter = make_adapter()
        command, execution = make_command(adapter, Ros2GoalState.EXECUTING)

        adapter._report_navigation_outcome(command)

        assert execution.finish_count == 0
        assert adapter.nav_command is command

    def test_an_unreadable_outcome_leaves_the_command_running(self):
        adapter = make_adapter()
        command, execution = make_command(adapter, Ros2GoalState.UNDECODABLE)

        adapter._report_navigation_outcome(command)

        assert execution.finish_count == 0
        assert adapter.nav_command is command
        assert adapter.node.logger.has('warn', 'could not read the outcome')

    def test_a_succeeded_goal_ends_the_command(self):
        adapter = make_adapter()
        command, execution = make_command(adapter, Ros2GoalState.SUCCEEDED)

        adapter._report_navigation_outcome(command)

        assert execution.finish_count == 1
        assert adapter.nav_command is None
        assert adapter.replan_counts == 0

    def test_a_canceled_goal_ends_the_command(self):
        adapter = make_adapter()
        command, execution = make_command(adapter, Ros2GoalState.CANCELED)

        adapter._report_navigation_outcome(command)

        assert execution.finish_count == 1
        assert adapter.nav_command is None

    def test_an_aborted_goal_raises_an_issue_and_replans(self):
        adapter = make_adapter()
        command, execution = make_command(adapter, Ros2GoalState.ABORTED)

        adapter._report_navigation_outcome(command)

        assert execution.finish_count == 0
        assert adapter.nav_command is None
        assert adapter.update_handle.more().issues
        assert adapter.update_handle.more().replan_count == 1

    def test_an_aborted_goal_is_only_ticketed_once(self):
        adapter = make_adapter()
        command, execution = make_command(adapter, Ros2GoalState.ABORTED)

        adapter._report_navigation_outcome(command)
        # The command is over, so a later tick has nothing left to report
        assert adapter.nav_command is None
        assert len(adapter.update_handle.more().issues) == 1

    def test_success_resolves_an_outstanding_issue_ticket(self):
        adapter = make_adapter()
        ticket = FakeIssueTicket()
        adapter.nav_issue_ticket = ticket
        command, execution = make_command(adapter, Ros2GoalState.SUCCEEDED)

        adapter._report_navigation_outcome(command)

        assert ticket.resolved
        assert adapter.nav_issue_ticket is None


if __name__ == '__main__':
    unittest.main()
