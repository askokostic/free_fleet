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

from dataclasses import dataclass, field

from free_fleet.ros2_types import (
    DockRobot_Feedback,
    DockRobot_GetResult_Request,
    DockRobot_GetResult_Response,
    DockRobot_SendGoal_Request,
    DockRobot_SendGoal_Response,
    GeometryMsgs_Point,
    GeometryMsgs_Pose,
    GeometryMsgs_PoseStamped,
    GeometryMsgs_Quaternion,
    Header,
    NavigateToPose_Feedback,
    NavigateToPose_GetResult_Request,
    NavigateToPose_GetResult_Response,
    NavigateToPose_SendGoal_Request,
    NavigateToPose_SendGoal_Response,
    Time,
)
from free_fleet_adapter.ros2_action_client import Ros2ActionCodec
from tf_transformations import quaternion_from_euler

NAVIGATE_TO_POSE_ACTION_NAME = 'navigate_to_pose'
DOCK_ROBOT_ACTION_NAME = 'dock_robot'


@dataclass
class NavigateToPoseGoal:
    """A pose for the robot to drive to, in a named frame."""

    frame_id: str
    x: float
    y: float
    z: float
    yaw: float
    stamp: Time = field(default_factory=lambda: Time(sec=0, nanosec=0))


class NavigateToPoseCodec(Ros2ActionCodec):
    """The wire shapes of the Nav2 NavigateToPose action."""

    def make_send_goal_request(self, goal_id, goal: NavigateToPoseGoal):
        quat = quaternion_from_euler(0, 0, goal.yaw)
        pose_stamped = GeometryMsgs_PoseStamped(
            header=Header(stamp=goal.stamp, frame_id=goal.frame_id),
            pose=GeometryMsgs_Pose(
                position=GeometryMsgs_Point(x=goal.x, y=goal.y, z=goal.z),
                orientation=GeometryMsgs_Quaternion(
                    x=quat[0],
                    y=quat[1],
                    z=quat[2],
                    w=quat[3]
                )
            )
        )

        return NavigateToPose_SendGoal_Request(
            goal_id=goal_id,
            pose=pose_stamped,
            behavior_tree=''
        )

    def decode_send_goal_response(self, payload):
        return NavigateToPose_SendGoal_Response.deserialize(payload)

    def make_get_result_request(self, goal_id):
        return NavigateToPose_GetResult_Request(goal_id=goal_id)

    def decode_get_result_response(self, payload):
        return NavigateToPose_GetResult_Response.deserialize(payload)

    def decode_feedback(self, payload):
        return NavigateToPose_Feedback.deserialize(payload)

    def estimated_remaining_sec(self, feedback) -> int | None:
        return feedback.estimated_time_remaining.sec


@dataclass
class DockRobotGoal:
    """A dock for the robot to drive into, named in the navigation graph."""

    dock_id: str


class DockRobotCodec(Ros2ActionCodec):
    """The wire shapes of the Nav2 DockRobot action."""

    USE_DOCK_ID = True
    # Since the dock ID is used, a dock pose is not required, but it needs to
    # be provided to satisfy the API
    DOCK_POSE = GeometryMsgs_PoseStamped(
        header=Header(stamp=Time(sec=0, nanosec=0), frame_id=''),
        pose=GeometryMsgs_Pose(
            position=GeometryMsgs_Point(x=0.0, y=0.0, z=0.0),
            orientation=GeometryMsgs_Quaternion()
        )
    )
    DOCK_TYPE = ''

    # The following are the default values in the DockRobot ROS 2 action
    MAX_STAGING_TIME = 1000.0
    NAVIGATE_TO_STAGING_POSE = True

    def make_send_goal_request(self, goal_id, goal: DockRobotGoal):
        return DockRobot_SendGoal_Request(
            goal_id=goal_id,
            dock_id=goal.dock_id,
            use_dock_id=self.USE_DOCK_ID,
            dock_pose=self.DOCK_POSE,
            dock_type=self.DOCK_TYPE,
            max_staging_time=self.MAX_STAGING_TIME,
            navigate_to_staging_pose=self.NAVIGATE_TO_STAGING_POSE,
        )

    def decode_send_goal_response(self, payload):
        return DockRobot_SendGoal_Response.deserialize(payload)

    def make_get_result_request(self, goal_id):
        return DockRobot_GetResult_Request(goal_id=goal_id)

    def decode_get_result_response(self, payload):
        return DockRobot_GetResult_Response.deserialize(payload)

    def decode_feedback(self, payload):
        return DockRobot_Feedback.deserialize(payload)
