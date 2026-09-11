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

"""Offline doubles standing in for a node, a zenoh session and an action type."""

import json

from free_fleet_adapter.ros2_action_client import Ros2ActionCodec


class FakeLogger:

    def __init__(self):
        self.messages = []

    def _record(self, level, msg):
        self.messages.append((level, str(msg)))

    def debug(self, msg):
        self._record('debug', msg)

    def info(self, msg):
        self._record('info', msg)

    def warn(self, msg):
        self._record('warn', msg)

    def error(self, msg):
        self._record('error', msg)

    def has(self, level, fragment):
        return any(level == lvl and fragment in msg
                   for lvl, msg in self.messages)


class FakeClock:

    def __init__(self, time_sec=1000):
        self.time_sec = time_sec

    def now(self):
        return self

    def seconds_nanoseconds(self):
        return (self.time_sec, 0)


class FakeNode:

    def __init__(self):
        self.logger = FakeLogger()
        self.clock = FakeClock()

    def get_logger(self):
        return self.logger

    def get_clock(self):
        return self.clock


class FakeZBytes:

    def __init__(self, payload: bytes):
        self.payload = payload

    def to_bytes(self) -> bytes:
        return self.payload

    def to_string(self) -> str:
        return self.payload.decode()


class FakeSample:

    def __init__(self, payload: bytes):
        self.payload = FakeZBytes(payload)


class FakeReply:

    def __init__(self, ok: bytes | None = None, err: bytes | None = None):
        self.ok = None if ok is None else FakeSample(ok)
        self.err = None if err is None else FakeSample(err)


class PendingQuery:
    """A callback query the fake session has accepted but not yet answered."""

    def __init__(self, key_expr, payload, handler):
        self.key_expr = key_expr
        self.payload = payload
        self.handler = handler

    def deliver(self, replies):
        for reply in replies:
            self.handler(reply)
        self.handler.drop()


class FakeZenohSession:
    """
    A zenoh session that answers from a script instead of from the network.

    Blocking gets are answered inline from `replies`, keyed by the suffix of the
    queried key expression. Callback gets are never answered inline: they are
    parked in `pending`, so a test can assert that the caller carried on without
    them and choose when the answer lands.
    """

    def __init__(self):
        self.subscribers = {}
        self.replies = {}
        self.queries = []
        self.pending = []
        self.observers = []

    def declare_subscriber(self, key_expr, handler):
        self.subscribers[key_expr] = handler
        return object()

    def publish(self, key_suffix, payload: bytes):
        for key_expr, handler in self.subscribers.items():
            if key_expr.endswith(key_suffix):
                handler(FakeSample(payload))

    def get(self, key_expr, handler=None, *, payload=None, timeout=None):
        self.queries.append((key_expr, payload))
        for observer in self.observers:
            observer(key_expr)
        if handler is None:
            return list(self._scripted_replies(key_expr))
        self.pending.append(PendingQuery(key_expr, payload, handler))
        return None

    def pending_for(self, key_suffix):
        return [query for query in self.pending
                if query.key_expr.endswith(key_suffix)]

    def answer_pending(self, key_suffix, replies):
        """Answer every parked query on key_suffix and forget it."""
        answered = self.pending_for(key_suffix)
        self.pending = [query for query in self.pending
                        if query not in answered]
        for query in answered:
            query.deliver(replies)
        return answered

    def queries_for(self, key_suffix):
        return [key for key, _ in self.queries if key.endswith(key_suffix)]

    def _scripted_replies(self, key_expr):
        for key_suffix, replies in self.replies.items():
            if key_expr.endswith(key_suffix):
                return replies
        return []


class FakeRequest:
    """A request that serializes to JSON rather than to CDR."""

    def __init__(self, **fields):
        self.fields = fields

    def serialize(self) -> bytes:
        return json.dumps(self.fields).encode()


class FakeMessage:

    def __init__(self, fields: dict):
        self.__dict__.update(fields)


class FakeCodec(Ros2ActionCodec):
    """
    A codec for an action type the adapter has never heard of.

    Messages travel as JSON, so a payload that is not JSON of the expected
    shape is the test's way of producing an undecodable reply.
    """

    def __init__(self, reports_estimate=False):
        self.reports_estimate = reports_estimate

    def make_send_goal_request(self, goal_id, goal):
        return FakeRequest(goal_id=list(goal_id), goal=goal)

    def decode_send_goal_response(self, payload):
        fields = json.loads(payload)
        return FakeMessage({'accepted': bool(fields['accepted'])})

    def make_get_result_request(self, goal_id):
        return FakeRequest(goal_id=list(goal_id))

    def decode_get_result_response(self, payload):
        fields = json.loads(payload)
        return FakeMessage({'status': int(fields['status'])})

    def decode_feedback(self, payload):
        fields = json.loads(payload)
        return FakeMessage({
            'goal_id': list(fields['goal_id']),
            'remaining_sec': int(fields.get('remaining_sec', 0)),
        })

    def estimated_remaining_sec(self, feedback):
        if not self.reports_estimate:
            return None
        return feedback.remaining_sec


def send_goal_response(accepted: bool) -> bytes:
    return json.dumps({'accepted': accepted}).encode()


def get_result_response(status: int) -> bytes:
    return json.dumps({'status': status}).encode()


def feedback_frame(goal_id, remaining_sec: int = 0) -> bytes:
    return json.dumps(
        {'goal_id': list(goal_id), 'remaining_sec': remaining_sec}).encode()
