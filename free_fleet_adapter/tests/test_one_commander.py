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
A robot takes commands from one place at a time.

Whatever is driving the robot holds a claim on it, and keeps that claim until
it reports that the robot has stopped - not until it is asked to stop, and not
merely for as long as one goal is in flight. A command arriving meanwhile asks
the holder to stop and waits for that report. The wait is bounded, because a
holder that never reports would otherwise strand every command after it.
"""

import unittest

from free_fleet_adapter.action import RobotActionState
from free_fleet_adapter.ros2_action_client import Ros2GoalState

from tests.adapter_doubles import (
    FakeAction,
    FakeActionFactory,
    FakeDestination,
    FakeExecution,
    give_the_robot_a_navigation,
    give_the_robot_an_action,
    make_adapter,
    wind_the_clock_on,
)


def another_action_arrives(adapter, category='some_action'):
    """Have RMF send the robot an action, and return it with its execution."""
    incoming = FakeAction()
    adapter.action_factories = {'plugin': FakeActionFactory(incoming)}
    execution = FakeExecution()
    adapter.execute_action(category, {}, execution)
    return incoming, execution


def a_navigation_arrives(adapter):
    """Have RMF send the robot somewhere, and return the execution for it."""
    execution = FakeExecution()
    adapter.navigate(FakeDestination(), execution)
    return execution


class TestARunningActionKeepsBeingUpdated(unittest.TestCase):

    def test_an_action_is_updated_while_a_navigation_waits_for_the_robot(self):
        # A plugin part-way through its work is still asked how it is doing
        adapter = make_adapter()
        action, _ = give_the_robot_an_action(adapter)
        a_navigation_arrives(adapter)

        adapter.update(None)
        adapter.update(None)

        assert action.updates == 2

    def test_a_navigation_never_runs_alongside_an_action(self):
        adapter = make_adapter()
        give_the_robot_an_action(adapter)

        a_navigation_arrives(adapter)

        assert adapter.nav_client.started == []

    def test_the_activity_reported_is_the_one_holding_the_robot(self):
        adapter = make_adapter()
        _, execution = give_the_robot_an_action(adapter)
        a_navigation_arrives(adapter)

        adapter.update(None)

        assert adapter.update_handle.reported == [execution.identifier]


class TestAskingTheHolderToStop(unittest.TestCase):

    def test_a_command_arriving_asks_the_holder_to_stop(self):
        adapter = make_adapter()
        action, _ = give_the_robot_an_action(adapter)

        another_action_arrives(adapter)

        assert action.stops == 1

    def test_the_holder_is_asked_only_once(self):
        adapter = make_adapter()
        action, _ = give_the_robot_an_action(adapter)

        another_action_arrives(adapter)
        adapter.update(None)
        adapter.update(None)

        assert action.stops == 1

    def test_the_incoming_command_waits_for_the_robot_to_stop(self):
        adapter = make_adapter()
        give_the_robot_an_action(adapter)

        incoming, _ = another_action_arrives(adapter)
        adapter.update(None)

        assert adapter.claim.exec_handle.action is not incoming
        assert incoming.updates == 0

    def test_the_incoming_command_is_admitted_once_the_robot_has_stopped(self):
        adapter = make_adapter()
        action, _ = give_the_robot_an_action(adapter)

        incoming, _ = another_action_arrives(adapter)
        action.state = RobotActionState.CANCELED
        adapter.update(None)

        assert adapter.claim.exec_handle.action is incoming

    def test_a_holder_winding_down_over_several_ticks_is_waited_out(self):
        # What a real stop looks like: the action reports CANCELING until the
        # endpoint confirms its goal is terminal, and only then CANCELED
        adapter = make_adapter()
        action, _ = give_the_robot_an_action(adapter)

        incoming, _ = another_action_arrives(adapter)
        action.state = RobotActionState.CANCELING
        for _ in range(4):
            adapter.update(None)
            assert adapter.claim.exec_handle.action is action

        action.state = RobotActionState.CANCELED
        adapter.update(None)

        assert adapter.claim.exec_handle.action is incoming
        assert adapter.update_handle.more().issues == []

    def test_a_command_arriving_at_a_free_robot_starts_at_once(self):
        adapter = make_adapter()

        incoming, _ = another_action_arrives(adapter)

        assert adapter.claim.exec_handle.action is incoming


class TestTheWaitForEvidenceIsBounded(unittest.TestCase):

    def test_a_holder_that_never_stops_does_not_strand_the_command(self):
        adapter = make_adapter(stop_timeout_sec=5.0)
        give_the_robot_an_action(adapter)

        incoming, _ = another_action_arrives(adapter)
        wind_the_clock_on(adapter, 6)
        adapter.update(None)

        assert adapter.claim.exec_handle.action is incoming

    def test_the_command_still_waits_inside_the_configured_interval(self):
        adapter = make_adapter(stop_timeout_sec=5.0)
        give_the_robot_an_action(adapter)

        incoming, _ = another_action_arrives(adapter)
        wind_the_clock_on(adapter, 4)
        adapter.update(None)

        assert adapter.claim.exec_handle.action is not incoming

    def test_the_interval_that_is_configured_is_the_one_that_is_waited(self):
        adapter = make_adapter(stop_timeout_sec=60.0)
        give_the_robot_an_action(adapter)

        incoming, _ = another_action_arrives(adapter)
        wind_the_clock_on(adapter, 30)
        adapter.update(None)

        assert adapter.claim.exec_handle.action is not incoming

    def test_an_expired_wait_raises_an_operator_visible_issue(self):
        adapter = make_adapter(stop_timeout_sec=5.0)
        give_the_robot_an_action(adapter)

        another_action_arrives(adapter)
        wind_the_clock_on(adapter, 6)
        adapter.update(None)

        assert len(adapter.update_handle.more().issues) == 1
        tier, category, detail = adapter.update_handle.more().issues[0]
        assert 'stop' in category

    def test_an_expired_wait_leaves_its_issue_open(self):
        # Dropping the ticket retracts the issue, so the operator sees nothing
        adapter = make_adapter(stop_timeout_sec=5.0)
        give_the_robot_an_action(adapter)

        another_action_arrives(adapter)
        wind_the_clock_on(adapter, 6)
        adapter.update(None)

        assert len(adapter.stop_timeout_issues) == 1
        _, ticket = adapter.stop_timeout_issues[0]
        assert not ticket.resolved

    def test_an_execution_taken_off_the_robot_is_not_released(self):
        # Dropping the last reference to it fires the same completion trigger
        # that finishing it would, so the claim is kept with its issue
        adapter = make_adapter(stop_timeout_sec=5.0)
        _, outgoing = give_the_robot_an_action(adapter)

        another_action_arrives(adapter)
        wind_the_clock_on(adapter, 6)
        adapter.update(None)

        held, _ = adapter.stop_timeout_issues[0]
        assert held.exec_handle.execution is outgoing

    def test_the_issue_is_raised_once_however_many_ticks_pass(self):
        adapter = make_adapter(stop_timeout_sec=5.0)
        give_the_robot_an_action(adapter)
        adapter.stop(adapter.claim.activity)

        wind_the_clock_on(adapter, 6)
        for _ in range(3):
            adapter.update(None)

        assert len(adapter.update_handle.more().issues) == 1

    def test_a_handover_with_evidence_raises_no_issue(self):
        adapter = make_adapter(stop_timeout_sec=5.0)
        action, _ = give_the_robot_an_action(adapter)

        another_action_arrives(adapter)
        action.state = RobotActionState.CANCELED
        adapter.update(None)

        assert adapter.update_handle.more().issues == []


class TestNothingIsQuietlyFinished(unittest.TestCase):

    def test_an_outgoing_execution_is_not_finished_to_make_room(self):
        adapter = make_adapter()
        _, outgoing = give_the_robot_an_action(adapter)

        another_action_arrives(adapter)

        assert outgoing.finish_count == 0

    def test_a_cancelled_action_is_never_reported_as_finished(self):
        adapter = make_adapter()
        action, outgoing = give_the_robot_an_action(adapter)

        another_action_arrives(adapter)
        action.state = RobotActionState.CANCELED
        adapter.update(None)

        assert outgoing.finish_count == 0

    def test_an_execution_admitted_over_is_not_finished_either(self):
        adapter = make_adapter(stop_timeout_sec=5.0)
        _, outgoing = give_the_robot_an_action(adapter)

        another_action_arrives(adapter)
        wind_the_clock_on(adapter, 6)
        adapter.update(None)

        assert outgoing.finish_count == 0

    def test_an_action_that_completes_still_reports_it(self):
        adapter = make_adapter()
        action, execution = give_the_robot_an_action(
            adapter, RobotActionState.COMPLETED)

        adapter.update(None)

        assert execution.finish_count == 1
        assert adapter.claim is None


class TestAPluginKeepsItsClaimBetweenItsGoals(unittest.TestCase):

    def test_a_navigation_does_not_slip_into_a_gap_between_goals(self):
        # The claim is the action, not whichever goal it happens to be running
        adapter = make_adapter()
        action, _ = give_the_robot_an_action(adapter)

        a_navigation_arrives(adapter)
        for _ in range(5):
            adapter.update(None)

        assert adapter.nav_client.started == []
        assert action.updates == 5

    def test_the_navigation_runs_once_the_action_is_over(self):
        adapter = make_adapter()
        action, _ = give_the_robot_an_action(adapter)

        a_navigation_arrives(adapter)
        action.state = RobotActionState.COMPLETED
        adapter.update(None)

        assert len(adapter.nav_client.started) == 1


class TestOneNavigationAtATime(unittest.TestCase):

    def test_an_incoming_navigation_cancels_the_one_in_flight(self):
        adapter = make_adapter()
        _, client, _ = give_the_robot_a_navigation(adapter)

        a_navigation_arrives(adapter)

        assert client.cancels == 1

    def test_the_replacement_waits_for_the_goal_to_be_terminal(self):
        adapter = make_adapter()
        _, client, _ = give_the_robot_a_navigation(adapter)

        a_navigation_arrives(adapter)
        adapter.update(None)

        assert client.started == []

    def test_the_replacement_starts_once_the_goal_is_terminal(self):
        adapter = make_adapter()
        _, client, _ = give_the_robot_a_navigation(adapter)

        a_navigation_arrives(adapter)
        client.reports(Ros2GoalState.CANCELED)
        adapter.update(None)

        assert len(client.started) == 1

    def test_a_superseded_navigation_is_not_reported_as_finished(self):
        # Finishing it would step the path RMF has already replaced
        adapter = make_adapter()
        _, client, execution = give_the_robot_a_navigation(adapter)

        a_navigation_arrives(adapter)
        client.reports(Ros2GoalState.CANCELED)
        adapter.update(None)

        assert execution.finish_count == 0

    def test_a_stopped_navigation_keeps_the_robot_until_its_goal_is_over(self):
        adapter = make_adapter()
        claim, client, _ = give_the_robot_a_navigation(adapter)

        adapter.stop(claim.activity)
        adapter.update(None)

        assert adapter.claim is claim
        assert client.cancels == 1

    def test_a_stopped_navigation_releases_the_robot_on_evidence(self):
        adapter = make_adapter()
        claim, client, _ = give_the_robot_a_navigation(adapter)

        adapter.stop(claim.activity)
        client.reports(Ros2GoalState.CANCELED)
        adapter.update(None)

        assert adapter.claim is None

    def test_an_unreadable_outcome_is_not_evidence_that_it_stopped(self):
        adapter = make_adapter()
        claim, client, _ = give_the_robot_a_navigation(adapter)

        adapter.stop(claim.activity)
        client.reports(Ros2GoalState.UNDECODABLE)
        adapter.update(None)

        assert adapter.claim is claim


class TestAfterAKill(unittest.TestCase):
    """
    A kill withdraws the execution and queues nothing behind it.

    So the action is asked to stop, it stops whichever goal is driving the
    robot, and the robot is released standing where it was when the kill
    landed. No command follows, and the adapter sends none of its own.
    """

    def test_a_killed_action_is_asked_to_stop(self):
        adapter = make_adapter()
        action, execution = give_the_robot_an_action(adapter)
        execution.rmf_takes_it_away()

        adapter.update(None)

        assert action.stops == 1

    def test_a_killed_action_keeps_the_robot_until_it_has_stopped(self):
        adapter = make_adapter()
        action, execution = give_the_robot_an_action(adapter)
        execution.rmf_takes_it_away()

        adapter.update(None)

        assert adapter.claim is not None
        assert action.updates == 1

    def test_a_killed_action_releases_the_robot_once_it_has(self):
        adapter = make_adapter()
        action, execution = give_the_robot_an_action(adapter)
        execution.rmf_takes_it_away()

        adapter.update(None)
        action.state = RobotActionState.CANCELED
        adapter.update(None)

        assert adapter.claim is None
        assert execution.finish_count == 0

    def test_a_holder_that_will_not_report_is_raised_with_nothing_waiting(
            self):
        # After a kill nothing is queued behind the robot, so there is no
        # handover to make - but an action that has stopped answering is
        # still something only an operator can deal with
        adapter = make_adapter(stop_timeout_sec=5.0)
        action, execution = give_the_robot_an_action(adapter)
        execution.rmf_takes_it_away()

        adapter.update(None)
        wind_the_clock_on(adapter, 6)
        adapter.update(None)

        assert len(adapter.update_handle.more().issues) == 1
        assert adapter.claim is not None
        assert action.updates == 2

    def test_nothing_commands_the_robot_after_a_kill(self):
        adapter = make_adapter()
        action, execution = give_the_robot_an_action(adapter)
        execution.rmf_takes_it_away()
        action.state = RobotActionState.CANCELED

        adapter.update(None)
        adapter.update(None)

        assert adapter.pending_command is None
        assert adapter.nav_client.started == []
        assert adapter.update_handle.more().issues == []


if __name__ == '__main__':
    unittest.main()
