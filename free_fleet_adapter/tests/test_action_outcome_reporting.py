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
"""

import unittest

from free_fleet_adapter.action import RobotActionState

from tests.adapter_doubles import (
    give_the_robot_an_action,
    make_adapter,
)


class TestReportingAnActionsOutcome(unittest.TestCase):

    def test_a_running_action_keeps_the_robot(self):
        adapter = make_adapter()
        action, execution = give_the_robot_an_action(adapter)

        adapter._report_action_outcome(adapter.claim)

        assert action.updates == 1
        assert execution.finish_count == 0
        assert adapter.claim is not None

    def test_an_action_winding_down_keeps_the_robot(self):
        adapter = make_adapter()
        action, execution = give_the_robot_an_action(
            adapter, RobotActionState.CANCELING)

        adapter._report_action_outcome(adapter.claim)

        assert execution.finish_count == 0
        assert adapter.claim is not None

    def test_a_completed_action_finishes_its_execution(self):
        adapter = make_adapter()
        action, execution = give_the_robot_an_action(
            adapter, RobotActionState.COMPLETED)

        adapter._report_action_outcome(adapter.claim)

        assert execution.finish_count == 1
        assert adapter.claim is None

    def test_a_failed_action_finishes_its_execution(self):
        adapter = make_adapter()
        action, execution = give_the_robot_an_action(
            adapter, RobotActionState.FAILED)

        adapter._report_action_outcome(adapter.claim)

        assert execution.finish_count == 1
        assert adapter.claim is None

    def test_a_cancelled_action_is_never_reported_as_finished(self):
        # RMF cancels the execution itself and gives the event its terminal
        # status; finishing it here would stamp Completed over that
        adapter = make_adapter()
        action, execution = give_the_robot_an_action(
            adapter, RobotActionState.CANCELED)

        adapter._report_action_outcome(adapter.claim)

        assert execution.finish_count == 0
        assert adapter.claim is None


class TestTellingAnActionRmfHasGivenUpOnIt(unittest.TestCase):

    def test_an_execution_rmf_withdrew_stops_the_action(self):
        adapter = make_adapter()
        action, execution = give_the_robot_an_action(adapter)
        execution.rmf_takes_it_away()

        adapter._report_action_outcome(adapter.claim)

        assert action.stops == 1

    def test_the_action_is_told_once_however_often_it_is_updated(self):
        adapter = make_adapter()
        action, execution = give_the_robot_an_action(adapter)
        execution.rmf_takes_it_away()

        adapter._report_action_outcome(adapter.claim)
        adapter._report_action_outcome(adapter.claim)
        adapter._report_action_outcome(adapter.claim)

        assert action.stops == 1

    def test_a_withdrawn_execution_is_still_updated_until_it_stops(self):
        # The robot's goal is still running, so the action keeps command of it
        adapter = make_adapter()
        action, execution = give_the_robot_an_action(adapter)
        execution.rmf_takes_it_away()

        adapter._report_action_outcome(adapter.claim)

        assert action.updates == 1
        assert adapter.claim is not None

    def test_an_action_rmf_still_wants_is_not_stopped(self):
        adapter = make_adapter()
        action, execution = give_the_robot_an_action(adapter)

        adapter._report_action_outcome(adapter.claim)

        assert action.stops == 0


class TestStoppingAnActivity(unittest.TestCase):

    def test_a_stop_reaches_the_running_action(self):
        adapter = make_adapter()
        action, execution = give_the_robot_an_action(adapter)

        adapter.stop(adapter.claim.activity)

        assert action.stops == 1

    def test_a_stopped_action_keeps_the_robot_until_it_says_it_stopped(self):
        # Dropping it here would leave its goal running, unwatched
        adapter = make_adapter()
        action, execution = give_the_robot_an_action(adapter)

        adapter.stop(adapter.claim.activity)

        assert adapter.claim is not None
        assert execution.finish_count == 0

    def test_a_stopped_action_is_told_once(self):
        adapter = make_adapter()
        action, execution = give_the_robot_an_action(adapter)

        adapter.stop(adapter.claim.activity)
        adapter.stop(adapter.claim.activity)

        assert action.stops == 1

    def test_a_stop_with_nothing_to_stop_is_harmless(self):
        adapter = make_adapter()

        adapter.stop(None)

        assert adapter.claim is None


if __name__ == '__main__':
    unittest.main()
