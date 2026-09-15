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

"""What the adapter reports to RMF about a navigation goal."""

import unittest

from free_fleet_adapter.ros2_action_client import Ros2GoalState

from tests.adapter_doubles import (
    FakeIssueTicket,
    give_the_robot_a_navigation,
    GOAL_ID,
    make_adapter,
    OTHER_GOAL_ID,
)


class TestReportingOnlyThisCommandsGoal(unittest.TestCase):

    def test_a_goal_that_has_not_been_sent_yet_is_not_reported(self):
        adapter = make_adapter()
        claim, client, execution = give_the_robot_a_navigation(
            adapter, Ros2GoalState.SUCCEEDED, command_goal_id=None)

        adapter._report_navigation_outcome(claim)

        assert execution.finish_count == 0
        assert adapter.claim is claim

    def test_another_goals_outcome_is_not_attributed_to_this_command(self):
        # The client is shared per endpoint, so it may still be reporting the
        # goal that ran before this command was given one
        adapter = make_adapter()
        claim, client, execution = give_the_robot_a_navigation(
            adapter,
            Ros2GoalState.SUCCEEDED,
            client_goal_id=OTHER_GOAL_ID,
            command_goal_id=GOAL_ID
        )

        adapter._report_navigation_outcome(claim)

        assert execution.finish_count == 0
        assert adapter.claim is claim

    def test_a_client_with_no_goal_does_not_complete_the_command(self):
        adapter = make_adapter()
        claim, client, execution = give_the_robot_a_navigation(
            adapter, Ros2GoalState.IDLE)

        adapter._report_navigation_outcome(claim)

        assert execution.finish_count == 0


class TestReportingAnOutcome(unittest.TestCase):

    def test_a_running_goal_leaves_the_command_alone(self):
        adapter = make_adapter()
        claim, client, execution = give_the_robot_a_navigation(
            adapter, Ros2GoalState.EXECUTING)

        adapter._report_navigation_outcome(claim)

        assert execution.finish_count == 0
        assert adapter.claim is claim

    def test_an_unreadable_outcome_leaves_the_command_running(self):
        adapter = make_adapter()
        claim, client, execution = give_the_robot_a_navigation(
            adapter, Ros2GoalState.UNDECODABLE)

        adapter._report_navigation_outcome(claim)

        assert execution.finish_count == 0
        assert adapter.claim is claim
        assert adapter.node.logger.has('warn', 'could not read the outcome')

    def test_a_succeeded_goal_ends_the_command(self):
        adapter = make_adapter()
        claim, client, execution = give_the_robot_a_navigation(
            adapter, Ros2GoalState.SUCCEEDED)

        adapter._report_navigation_outcome(claim)

        assert execution.finish_count == 1
        assert adapter.claim is None
        assert adapter.replan_counts == 0

    def test_a_canceled_goal_ends_the_command(self):
        adapter = make_adapter()
        claim, client, execution = give_the_robot_a_navigation(
            adapter, Ros2GoalState.CANCELED)

        adapter._report_navigation_outcome(claim)

        assert execution.finish_count == 1
        assert adapter.claim is None

    def test_an_aborted_goal_raises_an_issue_and_replans(self):
        adapter = make_adapter()
        claim, client, execution = give_the_robot_a_navigation(
            adapter, Ros2GoalState.ABORTED)

        adapter._report_navigation_outcome(claim)

        assert execution.finish_count == 0
        assert adapter.claim is None
        assert adapter.update_handle.more().issues
        assert adapter.update_handle.more().replan_count == 1

    def test_an_aborted_goal_is_only_ticketed_once(self):
        adapter = make_adapter()
        claim, client, execution = give_the_robot_a_navigation(
            adapter, Ros2GoalState.ABORTED)

        adapter._report_navigation_outcome(claim)
        # The command is over, so a later tick has nothing left to report
        assert adapter.claim is None
        assert len(adapter.update_handle.more().issues) == 1

    def test_success_resolves_an_outstanding_issue_ticket(self):
        adapter = make_adapter()
        ticket = FakeIssueTicket()
        adapter.nav_issue_ticket = ticket
        claim, client, execution = give_the_robot_a_navigation(
            adapter, Ros2GoalState.SUCCEEDED)

        adapter._report_navigation_outcome(claim)

        assert ticket.resolved
        assert adapter.nav_issue_ticket is None


class TestReportingOnAGoalThatWasAskedToStop(unittest.TestCase):
    """
    A goal whose stop the adapter asked for is no longer RMF's business.

    RMF is not waiting on a command it has already taken back, and finishing
    one would step a path RMF has replaced. The only thing still being waited
    on is that the goal has stopped driving the robot.
    """

    def test_a_stopped_goal_is_not_reported_as_finished(self):
        adapter = make_adapter()
        claim, client, execution = give_the_robot_a_navigation(
            adapter, Ros2GoalState.CANCELED)
        adapter.stop(claim.activity)

        adapter._report_navigation_outcome(claim)

        assert execution.finish_count == 0
        assert adapter.claim is None

    def test_a_stopped_goal_that_succeeded_anyway_is_not_reported_either(self):
        adapter = make_adapter()
        claim, client, execution = give_the_robot_a_navigation(
            adapter, Ros2GoalState.SUCCEEDED)
        adapter.stop(claim.activity)

        adapter._report_navigation_outcome(claim)

        assert execution.finish_count == 0
        assert adapter.claim is None

    def test_a_stopped_goal_that_aborted_does_not_replan(self):
        adapter = make_adapter()
        claim, client, execution = give_the_robot_a_navigation(
            adapter, Ros2GoalState.ABORTED)
        adapter.stop(claim.activity)

        adapter._report_navigation_outcome(claim)

        assert adapter.update_handle.more().replan_count == 0
        assert adapter.update_handle.more().issues == []
        assert adapter.claim is None


if __name__ == '__main__':
    unittest.main()
