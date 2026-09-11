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

import unittest

from free_fleet_adapter.nav2_action_codecs import (
    DOCK_ROBOT_ACTION_NAME,
    DockRobotCodec,
    DockRobotGoal,
    NAVIGATE_TO_POSE_ACTION_NAME,
    NavigateToPoseCodec,
    NavigateToPoseGoal,
)
from free_fleet_adapter.ros2_action_client import (
    RequestUnanswered,
    Ros2ActionClient,
    Ros2GoalState,
)
import rclpy

import zenoh


class TestNav2ActionClient(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('test_nav2_action_client')
        cls.zenoh_session = zenoh.open(zenoh.Config())

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        cls.zenoh_session.close()
        rclpy.shutdown()

    def make_client(self, action_name, codec):
        return Ros2ActionClient(
            node=self.node,
            zenoh_session=self.zenoh_session,
            robot_name='nav2_tb3',
            action_name=action_name,
            codec=codec,
            service_call_timeout_sec=1.0
        )

    def test_idle_navigate_to_pose_client_has_no_goal(self):
        client = self.make_client(
            NAVIGATE_TO_POSE_ACTION_NAME, NavigateToPoseCodec())

        assert client.goal_id is None
        assert client.outcome().state == Ros2GoalState.IDLE

    def test_idle_dock_client_has_no_goal(self):
        client = self.make_client(DOCK_ROBOT_ACTION_NAME, DockRobotCodec())

        assert client.goal_id is None
        assert client.outcome().state == Ros2GoalState.IDLE

    def test_cancelling_an_idle_navigate_to_pose_client_is_harmless(self):
        client = self.make_client(
            NAVIGATE_TO_POSE_ACTION_NAME, NavigateToPoseCodec())

        client.cancel()

        assert client.outcome().state == Ros2GoalState.IDLE

    def test_cancelling_an_idle_dock_client_is_harmless(self):
        client = self.make_client(DOCK_ROBOT_ACTION_NAME, DockRobotCodec())

        client.cancel()

        assert client.outcome().state == Ros2GoalState.IDLE

    def test_a_goal_sent_to_a_missing_endpoint_does_not_start(self):
        client = self.make_client(
            'no_such_action', NavigateToPoseCodec())

        with self.assertRaises(RequestUnanswered):
            client.start(NavigateToPoseGoal(
                frame_id='map', x=0.0, y=0.0, z=0.0, yaw=0.0))

        # Nobody refused it, so the goal stays named and cancellable
        assert client.goal_id is not None
        client.cancel()

    def test_a_dock_goal_sent_to_a_missing_endpoint_does_not_start(self):
        client = self.make_client('no_such_action', DockRobotCodec())

        with self.assertRaises(RequestUnanswered):
            client.start(DockRobotGoal(dock_id='dock'))

        assert client.goal_id is not None
        client.cancel()
