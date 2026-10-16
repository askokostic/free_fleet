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
The doubles the robot adapter's offline tests share.

Nav2RobotAdapter.__init__ waits for a live robot to localize, so make_adapter
builds the adapter without one and populates only the fields these paths read.
"""

from threading import Lock, RLock

from free_fleet_adapter.action import RobotActionState
from free_fleet_adapter.nav2_robot_adapter import (
    ActionClaim,
    Nav2RobotAdapter,
    NavigationClaim,
)
from free_fleet_adapter.robot_adapter import ExecutionHandle
from free_fleet_adapter.ros2_action_client import (
    Ros2GoalOutcome,
    Ros2GoalState,
)

from tests.fake_ros2_action_endpoint import FakeNode, FakeZenohSession

GOAL_ID = list(range(16))
OTHER_GOAL_ID = list(range(16, 32))


class FakeExecution:
    """The RMF execution a command reports to, and which RMF can withdraw."""

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


class FakeActionClient:
    """A client that reports a scripted outcome for a scripted goal."""

    def __init__(self, state, goal_id=GOAL_ID, detail='scripted'):
        self.scripted = Ros2GoalOutcome(state, goal_id, detail)
        self.cancels = 0
        self.started = []
        # What the endpoint answers to a cancel: the goal was already over
        self.goal_was_over = False

    def outcome(self):
        return self.scripted

    def cancel(self):
        self.cancels += 1
        return self.goal_was_over

    def start(self, goal):
        self.started.append(goal)
        return GOAL_ID

    def reports(self, state, goal_id=GOAL_ID, detail='scripted'):
        self.scripted = Ros2GoalOutcome(state, goal_id, detail)


class FakeDestination:
    """Where RMF wants the robot to go next."""

    def __init__(self, position=(1.0, 2.0, 0.0), map_name='L1', dock=None):
        self.position = list(position)
        self.map = map_name
        self.dock = dock


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
        self.reported = []

    def more(self):
        return self._more

    def update(self, state, activity_identifier):
        self.reported.append(activity_identifier)


def make_adapter(stop_timeout_sec=10.0, use_evacuation_monitor=False):
    adapter = Nav2RobotAdapter.__new__(Nav2RobotAdapter)
    adapter.name = 'robogreen'
    adapter.node = FakeNode()
    adapter.update_handle = FakeUpdateHandle()
    adapter.claim = None
    adapter.pending_command = None
    adapter.claim_mutex = RLock()
    adapter.stop_timeout_sec = stop_timeout_sec
    adapter.stop_timeout_issues = []
    adapter.action_to_plugin_name = {}
    adapter.action_factories = {}
    adapter.action_clients = {}
    adapter.nav_issue_ticket = None
    adapter.replan_counts = 0
    adapter.zenoh_session = FakeZenohSession()
    adapter.service_call_timeout_sec = 1.0
    adapter.use_evacuation_monitor = use_evacuation_monitor
    adapter.gate_mutex = Lock()
    adapter.evacuation_status = None
    adapter.evacuation_status_sec = None
    adapter.evacuation_monitor_had_data = False
    adapter.start_sec = adapter.node.clock.time_sec
    adapter.cancel_window_end_sec = None
    adapter.remote_hold = False
    adapter.gate_closed = False
    adapter.no_status_issue = None
    adapter.monitor_issue = None
    adapter.monitor_issue_state = None
    adapter.replan_on_gate_open = False
    # One clock for the stop wait and for the evacuation gate
    adapter._monotonic_sec = lambda: adapter.node.clock.time_sec
    adapter.map_name = 'L1'
    adapter.map_frame = 'map'
    adapter.nav_client = FakeActionClient(Ros2GoalState.EXECUTING)
    adapter.create_ros2_action_client = \
        lambda action_name, codec: adapter.nav_client
    return adapter


def give_the_robot_an_action(
    adapter,
    state=RobotActionState.IN_PROGRESS,
    okay=True,
    category='some_action'
):
    """Give the adapter a running action, and return it with its execution."""
    execution = FakeExecution(okay=okay)
    action = FakeAction(state)
    exec_handle = ExecutionHandle(execution)
    exec_handle.set_action(action)
    adapter.claim = ActionClaim(f'an action [{category}]', exec_handle)
    return action, execution


def give_the_robot_a_navigation(
    adapter,
    state=Ros2GoalState.EXECUTING,
    client_goal_id=GOAL_ID,
    command_goal_id=GOAL_ID
):
    """Give the adapter a navigation in flight, and return its parts."""
    execution = FakeExecution()
    exec_handle = ExecutionHandle(execution)
    exec_handle.set_goal_id(command_goal_id)
    client = adapter.nav_client
    client.reports(state, client_goal_id)
    adapter.claim = NavigationClaim('a navigation', client, exec_handle)
    return adapter.claim, client, execution


def wind_the_clock_on(adapter, seconds):
    """Let the robot's stop wait run for another `seconds`."""
    adapter.node.clock.time_sec += seconds
