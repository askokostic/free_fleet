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

from abc import ABC, abstractmethod
from enum import StrEnum
from threading import Lock

from free_fleet.ros2_types import (
    ActionMsgs_CancelGoal_Response,
    GoalStatus,
)
from free_fleet.utils import (
    make_nav2_cancel_all_goals_request,
    namespacify,
)
import numpy as np
import zenoh


class GoalNotAccepted(Exception):
    """A goal did not start, so nothing is running for the caller to await."""


class RequestRejected(GoalNotAccepted):
    """The action server refused the goal."""


class RequestUndecodable(GoalNotAccepted):
    """A reply did not match the wire shape its codec expects."""


class RequestUnanswered(GoalNotAccepted):
    """Nobody answered the request, so the goal may or may not be running."""


class Ros2GoalState(StrEnum):
    """What the client currently knows about the goal it last started."""

    IDLE = 'idle'
    EXECUTING = 'executing'
    SUCCEEDED = 'succeeded'
    CANCELED = 'canceled'
    ABORTED = 'aborted'
    UNDECODABLE = 'undecodable'


# States in which the goal is over. IDLE is deliberately absent: it means no
# goal was ever started, which is not the same as one that has finished.
TERMINAL_GOAL_STATES = frozenset({
    Ros2GoalState.SUCCEEDED,
    Ros2GoalState.CANCELED,
    Ros2GoalState.ABORTED,
})

GOAL_STATUS_STATES = {
    GoalStatus.STATUS_ACCEPTED.value: Ros2GoalState.EXECUTING,
    GoalStatus.STATUS_EXECUTING.value: Ros2GoalState.EXECUTING,
    GoalStatus.STATUS_CANCELING.value: Ros2GoalState.EXECUTING,
    GoalStatus.STATUS_SUCCEEDED.value: Ros2GoalState.SUCCEEDED,
    GoalStatus.STATUS_CANCELED.value: Ros2GoalState.CANCELED,
}


class Ros2ActionCodec(ABC):
    """
    The wire shapes of one ROS 2 action type.

    A codec is the only thing a Ros2ActionClient knows about the action it
    drives, so supplying one is all it takes to command an action the adapter
    has never heard of. Codecs hold no per-goal state: everything specific to a
    goal arrives as the opaque `goal` handed to Ros2ActionClient.start.
    """

    @abstractmethod
    def make_send_goal_request(self, goal_id, goal):
        """Return the serializable send_goal request for goal, as goal_id."""
        ...

    @abstractmethod
    def decode_send_goal_response(self, payload):
        """Return the send_goal response, which must carry `accepted`."""
        ...

    @abstractmethod
    def make_get_result_request(self, goal_id):
        """Return the serializable get_result request for goal_id."""
        ...

    @abstractmethod
    def decode_get_result_response(self, payload):
        """Return the get_result response, which must carry `status`."""
        ...

    @abstractmethod
    def decode_feedback(self, payload):
        """Return the feedback message, which must carry `goal_id`."""
        ...

    def estimated_remaining_sec(self, feedback) -> int | None:
        """
        Return how much longer the server expects the goal to need.

        Actions that report no estimate leave this as None, and the client asks
        for a result on every observation instead.
        """
        return None


class Ros2ActionFeedback:
    """The last feedback frame a goal produced, and when it was received."""

    def __init__(self, message, time_sec: int, estimated_remaining_sec):
        self.message = message
        self.time_sec = time_sec
        self.estimated_remaining_sec = estimated_remaining_sec

    def may_still_be_running(self, now_sec: int) -> bool:
        """Whether the server's own estimate says the goal cannot be over."""
        if self.estimated_remaining_sec is None:
            return False
        return self.time_sec + self.estimated_remaining_sec > now_sec


class Ros2GoalOutcome:
    """What the client knows about a goal, without asking the network."""

    def __init__(
        self,
        state: Ros2GoalState,
        goal_id=None,
        detail=None,
        result=None,
    ):
        self.state = state
        self.goal_id = goal_id
        self.detail = detail
        # The decoded get_result response, for actions that answer with more
        # than a status: a planner returns the plan there, and the caller has
        # no other way to read it. It is None until a result decodes, and it
        # belongs to goal_id like every other field here.
        self.result = result

    @property
    def is_done(self) -> bool:
        return self.state in TERMINAL_GOAL_STATES


class Ros2ActionClient:
    """
    Drives goals on one ROS 2 action endpoint, over zenoh.

    The endpoint name and every wire codec come from the caller, so the client
    carries no knowledge of what the action means. It also holds no reference to
    the RMF execution a goal belongs to: reporting outcomes to RMF is the robot
    adapter's job, and this client only reports them to the adapter.

    A client is bound to its endpoint for life rather than to a single goal, so
    its feedback subscription is declared once and successive goals reuse it.
    """

    def __init__(
        self,
        node,
        zenoh_session,
        robot_name: str,
        action_name: str,
        codec: Ros2ActionCodec,
        service_call_timeout_sec: float,
    ):
        self.node = node
        self.zenoh_session = zenoh_session
        self.robot_name = robot_name
        self.action_name = action_name
        self.codec = codec
        self.service_call_timeout_sec = service_call_timeout_sec

        self._mutex = Lock()
        self._goal_id = None
        self._state = Ros2GoalState.IDLE
        self._detail = None
        self._result = None
        self._feedback: Ros2ActionFeedback | None = None
        # Goal ID of the result query currently on the wire, if any
        self._result_query_goal_id = None

        self.feedback_sub = self.zenoh_session.declare_subscriber(
            self._endpoint('feedback'),
            self._receive_feedback
        )

    @property
    def goal_id(self):
        with self._mutex:
            return self._goal_id

    @property
    def last_feedback(self) -> Ros2ActionFeedback | None:
        with self._mutex:
            return self._feedback

    def start(self, goal):
        """
        Send goal to the endpoint and return the ID it was sent under.

        The goal ID is recorded before the request leaves, so a goal the server
        accepts is always one this client can name and cancel, even if the
        acceptance never makes it back.

        Raises GoalNotAccepted if the goal did not start: a refusal, an
        unreadable reply and silence are each reported as themselves, and none
        of them is ever read as an acceptance. Only an explicit refusal proves
        nothing is running, so only that case forgets the goal; after the other
        two the caller still holds a goal it can cancel.
        """
        goal_id = self._generate_goal_id()
        request = self.codec.make_send_goal_request(goal_id, goal)

        with self._mutex:
            self._goal_id = goal_id
            self._state = Ros2GoalState.EXECUTING
            self._detail = None
            self._result = None
            self._feedback = None
            self._result_query_goal_id = None

        errors = []
        for reply in self.zenoh_session.get(
            self._endpoint('send_goal'),
            payload=request.serialize(),
            timeout=self.service_call_timeout_sec
        ):
            if reply.ok is None:
                errors.append(self._error_of(reply))
                continue

            try:
                response = self.codec.decode_send_goal_response(
                    reply.ok.payload.to_bytes())
            except Exception as e:
                # The server may well have accepted a goal we cannot read the
                # answer to, so the goal stays named and cancellable
                raise RequestUndecodable(
                    f'{self.action_name} goal {goal_id} received a send_goal '
                    f'reply that does not decode: {type(e)}: {e}'
                ) from e

            if response.accepted:
                self.node.get_logger().info(
                    f'{self.action_name} goal {goal_id} accepted'
                )
                return goal_id

            self._forget_goal(goal_id)
            raise RequestRejected(
                f'{self.action_name} goal {goal_id} was rejected'
            )

        # Silence is not a refusal: the goal may be running, so it stays named
        raise RequestUnanswered(
            f'{self.action_name} goal {goal_id} was not answered'
            f'{": " + "; ".join(errors) if errors else ""}'
        )

    def outcome(self) -> Ros2GoalOutcome:
        """
        Return what is known about the running goal, from cached state alone.

        This never waits on the network, so it is safe to call from the
        adapter's update tick. When a fresh result is due, the query is put on
        the wire without being waited for, and its answer lands in the cache for
        a later call to read.
        """
        goal_id = None
        with self._mutex:
            outcome = Ros2GoalOutcome(
                self._state, self._goal_id, self._detail, self._result)
            query_due = (
                self._goal_id is not None and
                self._state not in TERMINAL_GOAL_STATES and
                self._result_query_goal_id is None and
                not self._feedback_defers_the_query()
            )
            if query_due:
                goal_id = self._goal_id
                self._result_query_goal_id = goal_id

        if goal_id is not None:
            self._request_result(goal_id)

        return outcome

    def cancel(self):
        """
        Ask the endpoint to cancel every goal it is running for this robot.

        The goal is not forgotten: its outcome still comes from the endpoint, so
        a cancellation is only reported once the endpoint confirms it.
        """
        with self._mutex:
            if self._goal_id is None:
                return
            # Drop the estimate so the next observation asks straight away
            self._feedback = None

        request = make_nav2_cancel_all_goals_request()
        for reply in self.zenoh_session.get(
            self._endpoint('cancel_goal'),
            payload=request.serialize(),
            timeout=self.service_call_timeout_sec
        ):
            if reply.ok is None:
                self.node.get_logger().error(
                    f'{self.action_name} cancel request for robot '
                    f'[{self.robot_name}] failed: {self._error_of(reply)}'
                )
                continue
            try:
                response = ActionMsgs_CancelGoal_Response.deserialize(
                    reply.ok.payload.to_bytes())
            except Exception as e:
                self.node.get_logger().error(
                    f'{self.action_name} cancel reply for robot '
                    f'[{self.robot_name}] does not decode: {type(e)}: {e}'
                )
                continue
            self.node.get_logger().info(
                f'{self.action_name} cancel return code: '
                f'{response.return_code}'
            )

    def _endpoint(self, service: str) -> str:
        return namespacify(
            f'{self.action_name}/_action/{service}', self.robot_name)

    def _generate_goal_id(self):
        return np.random.randint(0, 255, size=(16)).astype('uint8').tolist()

    def _now_sec(self) -> int:
        return self.node.get_clock().now().seconds_nanoseconds()[0]

    def _error_of(self, reply) -> str:
        if reply.err is None:
            return '[reply error not set]'
        return reply.err.payload.to_string()

    def _forget_goal(self, goal_id):
        with self._mutex:
            if self._goal_id != goal_id:
                return
            self._goal_id = None
            self._state = Ros2GoalState.IDLE
            self._detail = None
            self._result = None
            self._feedback = None
            self._result_query_goal_id = None

    def _feedback_defers_the_query(self) -> bool:
        """Whether the server's own estimate makes a result query pointless."""
        if self._feedback is None:
            return False
        return self._feedback.may_still_be_running(self._now_sec())

    def _request_result(self, goal_id):
        request = self.codec.make_get_result_request(goal_id)

        def _on_reply(reply):
            self._receive_result(goal_id, reply)

        def _on_query_done():
            with self._mutex:
                if self._result_query_goal_id == goal_id:
                    self._result_query_goal_id = None

        self.zenoh_session.get(
            self._endpoint('get_result'),
            zenoh.handlers.Callback(_on_reply, _on_query_done),
            payload=request.serialize(),
            timeout=self.service_call_timeout_sec
        )

    def _receive_result(self, goal_id, reply):
        if reply.ok is None:
            self.node.get_logger().debug(
                f'{self.action_name} result query for goal {goal_id} failed: '
                f'{self._error_of(reply)}'
            )
            return

        try:
            response = self.codec.decode_get_result_response(
                reply.ok.payload.to_bytes())
        except Exception as e:
            self._set_state(
                goal_id,
                Ros2GoalState.UNDECODABLE,
                f'{self.action_name} goal {goal_id} received a get_result '
                f'reply that does not decode: {type(e)}: {e}'
            )
            return

        state = GOAL_STATUS_STATES.get(response.status, Ros2GoalState.ABORTED)
        detail = None
        if state is Ros2GoalState.ABORTED:
            detail = (
                f'{self.action_name} goal {goal_id} ended with result status '
                f'[{response.status}]'
            )
        self._set_state(goal_id, state, detail, response)

    def _set_state(
        self,
        goal_id,
        state: Ros2GoalState,
        detail=None,
        result=None,
    ):
        """Record a state, unless it belongs to a goal that has been replaced."""
        with self._mutex:
            if self._goal_id != goal_id:
                return
            self._state = state
            self._detail = detail
            self._result = result

    def _receive_feedback(self, sample: zenoh.Sample):
        try:
            message = self.codec.decode_feedback(sample.payload.to_bytes())
            message_goal_id = list(message.goal_id)
        except Exception as e:
            self.node.get_logger().debug(
                f'{self.action_name} feedback for robot [{self.robot_name}] '
                f'does not decode: {type(e)}: {e}'
            )
            return

        feedback = Ros2ActionFeedback(
            message,
            self._now_sec(),
            self.codec.estimated_remaining_sec(message)
        )
        with self._mutex:
            if self._goal_id is None or message_goal_id != self._goal_id:
                return
            self._feedback = feedback
