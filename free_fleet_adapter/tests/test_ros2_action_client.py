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

import ast
import json
import unittest

from free_fleet.ros2_types import GoalStatus

import free_fleet_adapter.ros2_action_client as ros2_action_client
from free_fleet_adapter.ros2_action_client import (
    RequestRejected,
    RequestUnanswered,
    RequestUndecodable,
    Ros2ActionClient,
    Ros2GoalState,
)

from tests.fake_ros2_action_endpoint import (
    FakeCodec,
    FakeNode,
    FakeReply,
    FakeZenohSession,
    feedback_frame,
    get_result_response,
    send_goal_response,
)

SEND_GOAL = 'sweep_the_field/_action/send_goal'
GET_RESULT = 'sweep_the_field/_action/get_result'
CANCEL_GOAL = 'sweep_the_field/_action/cancel_goal'
FEEDBACK = 'sweep_the_field/_action/feedback'


def make_client(session, node=None, codec=None, action_name='sweep_the_field'):
    return Ros2ActionClient(
        node=node if node is not None else FakeNode(),
        zenoh_session=session,
        robot_name='robogreen',
        action_name=action_name,
        codec=codec if codec is not None else FakeCodec(),
        service_call_timeout_sec=1.0,
    )


def accepting_session():
    session = FakeZenohSession()
    session.replies[SEND_GOAL] = [FakeReply(ok=send_goal_response(True))]
    return session


def start_goal(session, **kwargs):
    client = make_client(session, **kwargs)
    goal_id = client.start({'field': 'north'})
    return client, goal_id


class TestArbitraryEndpoint(unittest.TestCase):
    """The client drives an action it knows nothing about."""

    def test_endpoints_are_derived_from_the_supplied_action_name(self):
        session = accepting_session()
        client, _ = start_goal(session)

        assert session.queries_for(f'robogreen/{SEND_GOAL}')
        assert f'robogreen/{FEEDBACK}' in session.subscribers
        assert client.action_name == 'sweep_the_field'

    def test_the_goal_travels_through_the_supplied_request_codec(self):
        session = accepting_session()
        _, goal_id = start_goal(session)

        _, payload = session.queries[0]
        assert json.loads(payload) == {
            'goal_id': list(goal_id), 'goal': {'field': 'north'}}

    def test_client_module_imports_no_domain_or_site_types(self):
        allowed = {
            'abc', 'enum', 'threading', 'typing',
            'free_fleet', 'numpy', 'zenoh',
        }
        with open(ros2_action_client.__file__, encoding='utf-8') as module:
            source = ast.parse(module.read())
        imported = set()
        for statement in ast.walk(source):
            if isinstance(statement, ast.Import):
                imported.update(
                    alias.name.split('.')[0] for alias in statement.names)
            elif isinstance(statement, ast.ImportFrom):
                imported.add((statement.module or '').split('.')[0])

        assert imported <= allowed, f'unexpected imports: {imported - allowed}'


class TestStartingAGoal(unittest.TestCase):

    def test_goal_identity_is_recorded_before_the_request_is_sent(self):
        session = accepting_session()
        client = make_client(session)
        seen = []
        session.observers.append(lambda key_expr: seen.append(client.goal_id))

        goal_id = client.start({'field': 'north'})

        assert seen == [goal_id]

    def test_an_accepted_goal_is_executing(self):
        session = accepting_session()
        client, goal_id = start_goal(session)

        outcome = client.outcome()
        assert outcome.state == Ros2GoalState.EXECUTING
        assert outcome.goal_id == goal_id

    def test_a_refused_goal_is_reported_as_rejected(self):
        session = FakeZenohSession()
        session.replies[SEND_GOAL] = [FakeReply(ok=send_goal_response(False))]
        client = make_client(session)

        with self.assertRaises(RequestRejected):
            client.start({'field': 'north'})

    def test_an_unreadable_reply_is_reported_as_undecodable(self):
        session = FakeZenohSession()
        session.replies[SEND_GOAL] = [FakeReply(ok=b'not the expected shape')]
        client = make_client(session)

        with self.assertRaises(RequestUndecodable):
            client.start({'field': 'north'})

    def test_an_unanswered_request_is_not_a_confident_acceptance(self):
        session = FakeZenohSession()
        client = make_client(session)

        with self.assertRaises(RequestUnanswered):
            client.start({'field': 'north'})

    def test_an_error_reply_is_not_a_confident_acceptance(self):
        session = FakeZenohSession()
        session.replies[SEND_GOAL] = [FakeReply(err=b'no such queryable')]
        client = make_client(session)

        with self.assertRaises(RequestUnanswered):
            client.start({'field': 'north'})

    def test_a_refused_goal_leaves_nothing_behind(self):
        session = FakeZenohSession()
        session.replies[SEND_GOAL] = [FakeReply(ok=send_goal_response(False))]
        client = make_client(session)

        with self.assertRaises(RequestRejected):
            client.start({'field': 'north'})

        assert client.goal_id is None
        assert client.outcome().state == Ros2GoalState.IDLE

    def test_an_unreadable_reply_leaves_a_goal_that_can_be_cancelled(self):
        session = FakeZenohSession()
        session.replies[SEND_GOAL] = [FakeReply(ok=b'not the expected shape')]
        client = make_client(session)

        with self.assertRaises(RequestUndecodable):
            client.start({'field': 'north'})

        # The server may have accepted it, so it must still be stoppable
        assert client.goal_id is not None
        client.cancel()
        assert session.queries_for(CANCEL_GOAL)

    def test_an_unanswered_request_leaves_a_goal_that_can_be_cancelled(self):
        session = FakeZenohSession()
        client = make_client(session)

        with self.assertRaises(RequestUnanswered):
            client.start({'field': 'north'})

        assert client.goal_id is not None
        client.cancel()
        assert session.queries_for(CANCEL_GOAL)


class TestObservingAGoal(unittest.TestCase):

    def test_an_idle_client_has_no_goal_to_report(self):
        session = FakeZenohSession()
        client = make_client(session)

        assert client.outcome().state == Ros2GoalState.IDLE
        assert not session.queries_for(GET_RESULT)

    def test_an_observation_never_waits_for_the_result_query(self):
        session = accepting_session()
        client, _ = start_goal(session)

        # The result query is parked, so the only way outcome() can return is
        # by answering from cached state.
        assert client.outcome().state == Ros2GoalState.EXECUTING
        assert len(session.pending_for(GET_RESULT)) == 1

    def test_only_one_result_query_is_in_flight_at_a_time(self):
        session = accepting_session()
        client, _ = start_goal(session)

        for _ in range(5):
            client.outcome()

        assert len(session.queries_for(GET_RESULT)) == 1

    def test_a_succeeded_result_is_reported_once_it_lands(self):
        session = accepting_session()
        client, _ = start_goal(session)
        client.outcome()

        session.answer_pending(GET_RESULT, [FakeReply(
            ok=get_result_response(GoalStatus.STATUS_SUCCEEDED.value))])

        assert client.outcome().state == Ros2GoalState.SUCCEEDED

    def test_a_canceled_result_is_reported_once_it_lands(self):
        session = accepting_session()
        client, _ = start_goal(session)
        client.outcome()

        session.answer_pending(GET_RESULT, [FakeReply(
            ok=get_result_response(GoalStatus.STATUS_CANCELED.value))])

        assert client.outcome().state == Ros2GoalState.CANCELED

    def test_an_aborted_result_carries_the_status_it_reported(self):
        session = accepting_session()
        client, _ = start_goal(session)
        client.outcome()

        session.answer_pending(GET_RESULT, [FakeReply(
            ok=get_result_response(GoalStatus.STATUS_ABORTED.value))])

        outcome = client.outcome()
        assert outcome.state == Ros2GoalState.ABORTED
        assert str(GoalStatus.STATUS_ABORTED.value) in outcome.detail

    def test_an_in_progress_result_keeps_the_goal_executing(self):
        session = accepting_session()
        client, _ = start_goal(session)
        client.outcome()

        session.answer_pending(GET_RESULT, [FakeReply(
            ok=get_result_response(GoalStatus.STATUS_EXECUTING.value))])

        assert client.outcome().state == Ros2GoalState.EXECUTING

    def test_an_unreadable_result_reply_is_reported_as_undecodable(self):
        session = accepting_session()
        client, _ = start_goal(session)
        client.outcome()

        session.answer_pending(GET_RESULT, [FakeReply(ok=b'garbage')])

        outcome = client.outcome()
        assert outcome.state == Ros2GoalState.UNDECODABLE
        assert outcome.detail

    def test_an_undecodable_result_does_not_end_the_goal(self):
        session = accepting_session()
        client, _ = start_goal(session)
        client.outcome()
        session.answer_pending(GET_RESULT, [FakeReply(ok=b'garbage')])

        client.outcome()
        session.answer_pending(GET_RESULT, [FakeReply(
            ok=get_result_response(GoalStatus.STATUS_SUCCEEDED.value))])

        assert client.outcome().state == Ros2GoalState.SUCCEEDED

    def test_a_terminal_goal_is_no_longer_queried(self):
        session = accepting_session()
        client, _ = start_goal(session)
        client.outcome()
        session.answer_pending(GET_RESULT, [FakeReply(
            ok=get_result_response(GoalStatus.STATUS_SUCCEEDED.value))])

        client.outcome()

        assert len(session.queries_for(GET_RESULT)) == 1

    def test_a_result_reply_is_ignored_once_a_new_goal_has_started(self):
        session = accepting_session()
        client, _ = start_goal(session)
        client.outcome()
        stale = session.pending_for(GET_RESULT)
        session.pending = []

        second_goal_id = client.start({'field': 'south'})
        for query in stale:
            query.deliver([FakeReply(
                ok=get_result_response(GoalStatus.STATUS_SUCCEEDED.value))])

        outcome = client.outcome()
        assert outcome.goal_id == second_goal_id
        assert outcome.state == Ros2GoalState.EXECUTING


class TestFeedback(unittest.TestCase):

    def test_feedback_for_the_running_goal_is_cached(self):
        session = accepting_session()
        client, goal_id = start_goal(session)

        session.publish(FEEDBACK, feedback_frame(goal_id, remaining_sec=12))

        assert client.last_feedback is not None
        assert client.last_feedback.message.remaining_sec == 12

    def test_feedback_for_another_goal_is_not_attributed_to_this_one(self):
        session = accepting_session()
        client, goal_id = start_goal(session)
        other_goal_id = [(byte + 1) % 256 for byte in goal_id]

        session.publish(FEEDBACK, feedback_frame(other_goal_id))

        assert client.last_feedback is None

    def test_unreadable_feedback_is_dropped_rather_than_cached(self):
        session = accepting_session()
        node = FakeNode()
        client, _ = start_goal(session, node=node)

        session.publish(FEEDBACK, b'garbage')

        assert client.last_feedback is None
        assert client.outcome().state == Ros2GoalState.EXECUTING

    def test_feedback_is_dropped_when_no_goal_is_running(self):
        session = FakeZenohSession()
        client = make_client(session)

        session.publish(FEEDBACK, feedback_frame(list(range(16))))

        assert client.last_feedback is None

    def test_a_reported_estimate_defers_the_next_result_query(self):
        session = accepting_session()
        node = FakeNode()
        client, goal_id = start_goal(
            session, node=node, codec=FakeCodec(reports_estimate=True))
        client.outcome()
        session.answer_pending(GET_RESULT, [FakeReply(
            ok=get_result_response(GoalStatus.STATUS_EXECUTING.value))])

        session.publish(FEEDBACK, feedback_frame(goal_id, remaining_sec=30))
        client.outcome()
        assert len(session.queries_for(GET_RESULT)) == 1

        node.clock.time_sec += 31
        client.outcome()
        assert len(session.queries_for(GET_RESULT)) == 2

    def test_a_goal_starts_without_the_previous_goals_feedback(self):
        session = accepting_session()
        client, goal_id = start_goal(session)
        session.publish(FEEDBACK, feedback_frame(goal_id, remaining_sec=12))

        client.start({'field': 'south'})

        assert client.last_feedback is None


class TestCancelling(unittest.TestCase):

    def test_cancelling_asks_the_endpoint_to_cancel(self):
        session = accepting_session()
        client, _ = start_goal(session)

        client.cancel()

        assert session.queries_for(CANCEL_GOAL)

    def test_an_idle_client_has_nothing_to_cancel(self):
        session = FakeZenohSession()
        client = make_client(session)

        client.cancel()

        assert not session.queries_for(CANCEL_GOAL)

    def test_the_outcome_of_a_cancelled_goal_still_comes_from_the_endpoint(self):
        session = accepting_session()
        client, goal_id = start_goal(session)

        client.cancel()
        client.outcome()
        session.answer_pending(GET_RESULT, [FakeReply(
            ok=get_result_response(GoalStatus.STATUS_CANCELED.value))])

        outcome = client.outcome()
        assert outcome.state == Ros2GoalState.CANCELED
        assert outcome.goal_id == goal_id


class TestWhatTheResultCarried(unittest.TestCase):
    """
    Some actions answer with more than a status, and the caller needs it.

    A planner's whole answer is its result, and the client is the only thing
    that ever sees one, so the decoded result travels on the outcome beside the
    state - and, like everything else there, it belongs to a named goal.
    """

    def test_a_result_is_handed_back_to_whoever_is_watching_the_goal(self):
        session = accepting_session()
        client, _ = start_goal(session)
        client.outcome()

        session.answer_pending(GET_RESULT, [FakeReply(
            ok=get_result_response(
                GoalStatus.STATUS_SUCCEEDED.value, plan=[1, 2, 3]))])

        assert client.outcome().result.plan == [1, 2, 3]

    def test_a_goal_that_has_not_answered_yet_carries_no_result(self):
        session = accepting_session()
        client, _ = start_goal(session)

        assert client.outcome().result is None

    def test_a_result_that_did_not_decode_is_no_result_at_all(self):
        session = accepting_session()
        client, _ = start_goal(session)
        client.outcome()

        session.answer_pending(GET_RESULT, [FakeReply(ok=b'garbage')])

        outcome = client.outcome()
        assert outcome.state == Ros2GoalState.UNDECODABLE
        assert outcome.result is None

    def test_a_new_goal_does_not_inherit_the_last_ones_result(self):
        session = accepting_session()
        client, _ = start_goal(session)
        client.outcome()
        session.answer_pending(GET_RESULT, [FakeReply(
            ok=get_result_response(
                GoalStatus.STATUS_SUCCEEDED.value, plan=[1, 2, 3]))])

        client.start({'field': 'south'})

        assert client.outcome().result is None


if __name__ == '__main__':
    unittest.main()
