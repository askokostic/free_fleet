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

import math
import unittest

from free_fleet.ros2_types import (
    DockRobot_Feedback,
    DockRobot_GetResult_Request,
    DockRobot_SendGoal_Request,
    DockRobot_SendGoal_Response,
    Duration,
    GeometryMsgs_Point,
    GeometryMsgs_Pose,
    GeometryMsgs_PoseStamped,
    GeometryMsgs_Quaternion,
    GoalStatus,
    Header,
    NavigateToPose_Feedback,
    NavigateToPose_GetResult_Request,
    NavigateToPose_GetResult_Response,
    NavigateToPose_SendGoal_Request,
    NavigateToPose_SendGoal_Response,
    Time,
)
from free_fleet_adapter.nav2_action_codecs import (
    DockRobotCodec,
    DockRobotGoal,
    NavigateToPoseCodec,
    NavigateToPoseGoal,
)
from tf_transformations import euler_from_quaternion

GOAL_ID = list(range(16))


class TestNavigateToPoseCodec(unittest.TestCase):

    def setUp(self):
        self.codec = NavigateToPoseCodec()

    def test_the_goal_pose_survives_the_wire(self):
        goal = NavigateToPoseGoal(
            frame_id='map',
            x=1.5,
            y=-2.5,
            z=0.25,
            yaw=math.pi / 2,
            stamp=Time(sec=7, nanosec=8),
        )

        request = NavigateToPose_SendGoal_Request.deserialize(
            self.codec.make_send_goal_request(GOAL_ID, goal).serialize())

        assert list(request.goal_id) == GOAL_ID
        assert request.pose.header.frame_id == 'map'
        assert (request.pose.header.stamp.sec,
                request.pose.header.stamp.nanosec) == (7, 8)
        assert math.isclose(request.pose.pose.position.x, 1.5)
        assert math.isclose(request.pose.pose.position.y, -2.5)
        assert math.isclose(request.pose.pose.position.z, 0.25)
        _, _, yaw = euler_from_quaternion([
            request.pose.pose.orientation.x,
            request.pose.pose.orientation.y,
            request.pose.pose.orientation.z,
            request.pose.pose.orientation.w,
        ])
        assert math.isclose(yaw, math.pi / 2)
        assert request.behavior_tree == ''

    def test_a_goal_without_a_stamp_is_stamped_zero(self):
        goal = NavigateToPoseGoal(frame_id='map', x=0.0, y=0.0, z=0.0, yaw=0.0)

        request = NavigateToPose_SendGoal_Request.deserialize(
            self.codec.make_send_goal_request(GOAL_ID, goal).serialize())

        assert (request.pose.header.stamp.sec,
                request.pose.header.stamp.nanosec) == (0, 0)

    def test_the_acceptance_reply_is_decoded(self):
        payload = NavigateToPose_SendGoal_Response(
            accepted=True, stamp=Time(sec=0, nanosec=0)).serialize()

        assert self.codec.decode_send_goal_response(payload).accepted

    def test_the_result_request_carries_the_goal_id(self):
        request = NavigateToPose_GetResult_Request.deserialize(
            self.codec.make_get_result_request(GOAL_ID).serialize())

        assert list(request.goal_id) == GOAL_ID

    def test_the_result_reply_is_decoded(self):
        payload = NavigateToPose_GetResult_Response(
            status=GoalStatus.STATUS_SUCCEEDED.value).serialize()

        response = self.codec.decode_get_result_response(payload)
        assert response.status == GoalStatus.STATUS_SUCCEEDED.value

    def test_feedback_is_decoded_and_reports_its_own_estimate(self):
        payload = _navigate_to_pose_feedback(
            estimated_time_remaining_sec=42).serialize()

        feedback = self.codec.decode_feedback(payload)

        assert list(feedback.goal_id) == GOAL_ID
        assert self.codec.estimated_remaining_sec(feedback) == 42


class TestDockRobotCodec(unittest.TestCase):

    def setUp(self):
        self.codec = DockRobotCodec()

    def test_the_dock_is_named_rather_than_posed(self):
        request = DockRobot_SendGoal_Request.deserialize(
            self.codec.make_send_goal_request(
                GOAL_ID, DockRobotGoal(dock_id='charger')).serialize())

        assert list(request.goal_id) == GOAL_ID
        assert request.dock_id == 'charger'
        assert request.use_dock_id
        assert request.dock_type == ''
        assert request.navigate_to_staging_pose
        assert math.isclose(request.max_staging_time, 1000.0)

    def test_the_acceptance_reply_is_decoded(self):
        payload = DockRobot_SendGoal_Response(
            accepted=False, stamp=Time(sec=0, nanosec=0)).serialize()

        assert not self.codec.decode_send_goal_response(payload).accepted

    def test_the_result_request_carries_the_goal_id(self):
        request = DockRobot_GetResult_Request.deserialize(
            self.codec.make_get_result_request(GOAL_ID).serialize())

        assert list(request.goal_id) == GOAL_ID

    def test_docking_feedback_reports_no_estimate(self):
        payload = DockRobot_Feedback(
            goal_id=GOAL_ID,
            state=0,
            docking_time=Duration(sec=3, nanosec=0),
            num_retries=0,
        ).serialize()

        feedback = self.codec.decode_feedback(payload)

        assert list(feedback.goal_id) == GOAL_ID
        assert self.codec.estimated_remaining_sec(feedback) is None


def _navigate_to_pose_feedback(estimated_time_remaining_sec: int):
    pose = GeometryMsgs_PoseStamped(
        header=Header(stamp=Time(sec=0, nanosec=0), frame_id='map'),
        pose=GeometryMsgs_Pose(
            position=GeometryMsgs_Point(x=0.0, y=0.0, z=0.0),
            orientation=GeometryMsgs_Quaternion(),
        ),
    )
    return NavigateToPose_Feedback(
        goal_id=GOAL_ID,
        current_pose=pose,
        navigation_time=Duration(sec=1, nanosec=0),
        estimated_time_remaining=Duration(
            sec=estimated_time_remaining_sec, nanosec=0),
        number_of_recoveries=0,
        distance_remaining=1.0,
    )


if __name__ == '__main__':
    unittest.main()
