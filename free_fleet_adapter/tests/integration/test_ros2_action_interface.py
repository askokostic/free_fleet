#!/usr/bin/env python3

# Copyright 2026 Open Source Robotics Foundation, Inc.
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

from free_fleet_adapter.robot_adapter import ExecutionHandle
from free_fleet_adapter.ros2_action_interface import (
    DockActionInterface,
    NavigateToPoseActionInterface
)
import rclpy

import zenoh


class TestRosActionInterface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('test_ros2_action_interface')
        cls.zenoh_session = zenoh.open(zenoh.Config())

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        cls.zenoh_session.close()
        rclpy.shutdown()

    def test_idle_robot_navigate_to_pose_is_done(self):
        exec_handle = ExecutionHandle(None)
        exec_handle.set_goal_id(None)  # Release mutex

        nav_handle = NavigateToPoseActionInterface(
            robot_name='nav2_tb3',
            node=self.node,
            update_handle=None,
            zenoh_session=self.zenoh_session,
            exec_handle=exec_handle,
            service_timeout=1.0,
            map_frame='map',
            x=0.0,
            y=0.0,
            z=0.0,
            yaw=0.0
        )

        is_done, _ = nav_handle.is_done()
        assert is_done

    def test_idle_robot_dock_is_done(self):
        exec_handle = ExecutionHandle(None)
        exec_handle.set_goal_id(None)  # Release mutex

        dock_handle = DockActionInterface(
            robot_name='nav2_tb3',
            node=self.node,
            update_handle=None,
            zenoh_session=self.zenoh_session,
            exec_handle=exec_handle,
            dock_id='dock',
            service_timeout=1.0
        )

        is_done, _ = dock_handle.is_done()
        assert is_done

    def test_stop_robot_navigate_to_pose_without_command(self):
        exec_handle = ExecutionHandle(None)
        exec_handle.set_goal_id(None)  # Release mutex

        nav_handle = NavigateToPoseActionInterface(
            robot_name='nav2_tb3',
            node=self.node,
            update_handle=None,
            zenoh_session=self.zenoh_session,
            exec_handle=exec_handle,
            service_timeout=1.0,
            map_frame='map',
            x=0.0,
            y=0.0,
            z=0.0,
            yaw=0.0
        )

        assert nav_handle.exec_handle.execution is None
        nav_handle.stop()
        assert nav_handle.exec_handle.execution is None

        is_done, _ = nav_handle.is_done()
        assert is_done

    def test_stop_robot_dock_without_command(self):
        exec_handle = ExecutionHandle(None)
        exec_handle.set_goal_id(None)  # Release mutex

        dock_handle = DockActionInterface(
            robot_name='nav2_tb3',
            node=self.node,
            update_handle=None,
            zenoh_session=self.zenoh_session,
            exec_handle=exec_handle,
            dock_id='dock',
            service_timeout=1.0
        )

        assert dock_handle.exec_handle.execution is None
        dock_handle.stop()
        assert dock_handle.exec_handle.execution is None

        is_done, _ = dock_handle.is_done()
        assert is_done
