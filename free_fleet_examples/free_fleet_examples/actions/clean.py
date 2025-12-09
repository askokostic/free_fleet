#!/usr/bin/env python3

import sys
import time
from typing import Optional, List
import argparse
import numpy as np
import zenoh

from free_fleet.ros2_types import (
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
from free_fleet.utils import namespacify

from free_fleet_adapter.action import (
    RobotAction,
    RobotActionContext,
    RobotActionFactory,
    RobotActionState,
)

class Clean(RobotAction):
    """A clean action that navigates through a path of target poses."""

    def __init__(self, description: dict, execution, context: RobotActionContext):
        super().__init__(context, execution)
        self.description = description
        self.state = RobotActionState.IN_PROGRESS
        self.namespace = self.context.robot_name
        self.goal_id = None
        self.session = None
        self.feedback_sub = None
        self.navigation_started = False
        self.result_received = False

        self.clean_path = self.context.action_config.get('clean_path', {}).get('path', [])
        self.current_index = 0

        if not self.clean_path:
            self.context.node.get_logger().error("No path defined for clean action.")
        else:
            # x, y = self.clean_path[0]
            # self.context.node.get_logger().info(f"Action started. First target: ({x}, {y})")
            self.context.node.get_logger().info("Path points for cleaning action:")
            for idx, (x, y) in enumerate(self.clean_path):
                self.context.node.get_logger().info(f"  {idx+1}: ({x}, {y})")

    def feedback_callback(self,sample: zenoh.Sample):
        feedback = NavigateToPose_Feedback.deserialize(sample.payload.to_bytes())
        # print(f'Distance remaining: {feedback.distance_remaining}')

    def run_navigate_to_pose_client(
        self,
        x: float,
        y: float,
        zenoh_config: Optional[str] = None,
        namespace: str = '',
        frame_id: str = 'map',
        timeout_sec: float = 5.5,
    ) -> Optional[int]:

        conf = zenoh.Config.from_file(zenoh_config) if zenoh_config else zenoh.Config()
        try:
            session = zenoh.open(conf)
        except Exception as e:
            self.context.node.get_logger().info(f"error: {e}")
            return None

        final_status = None

        feedback_sub = session.declare_subscriber(
            namespacify('navigate_to_pose/_action/feedback', namespace),
            self.feedback_callback
        )

        stamp = Time(sec=0, nanosec=0)
        header = Header(stamp=stamp, frame_id=frame_id)
        position = GeometryMsgs_Point(x=x, y=y, z=0)
        orientation = GeometryMsgs_Quaternion()
        pose = GeometryMsgs_Pose(position=position, orientation=orientation)
        pose_stamped = GeometryMsgs_PoseStamped(header=header, pose=pose)
        self.context.node.get_logger().info(str(pose_stamped))
        goal_id: List[int] = np.random.randint(0, 255, size=(16)).astype('uint8').tolist()
        self.context.node.get_logger().info(f"Goal ID: {goal_id}")

        req = NavigateToPose_SendGoal_Request(
            goal_id=goal_id,
            pose=pose_stamped,
            behavior_tree=''
        )

        replies = session.get(
            namespacify('navigate_to_pose/_action/send_goal', namespace),
            payload=req.serialize(),
        )

        for reply in replies:
            if not reply.ok:
                self.context.node.get_logger().info('Reply was not ok!')
                continue
            self.context.node.get_logger().info('handling a reply!')
            # Deserialize the response
            rep = NavigateToPose_SendGoal_Response.deserialize(
                reply.ok.payload.to_bytes()
            )
            if not rep.accepted:
                self.context.node.get_logger().info('Goal rejected')
                return

        self.context.node.get_logger().info('Goal accepted by server, waiting for result')
        req = NavigateToPose_GetResult_Request(goal_id)

        try:
            while True:
                # Send the query with the serialized request
                replies = session.get(
                    namespacify(
                        'navigate_to_pose/_action/get_result',
                        namespace),
                    payload=req.serialize(),
                    timeout=5.5
                )

                for reply in replies:
                    try:
                        if not reply.ok:
                            self.context.node.get_logger().info('Reply was not ok!')
                            continue
                        # Deserialize the response
                        rep = NavigateToPose_GetResult_Response.deserialize(
                            reply.ok.payload.to_bytes()
                        )
                        self.context.node.get_logger().info(f'Result: {rep.status}')
                        if rep.status == GoalStatus.STATUS_ABORTED.value:
                            self.context.node.get_logger().info(
                                'Received STATUS_ABORTED'
                            )
                            final_status = rep.status
                            return final_status
                            # break
                        if rep.status == GoalStatus.STATUS_SUCCEEDED.value:
                            self.context.node.get_logger().info(
                                'Received STATUS_SUCCEEDED'
                            )
                            final_status = rep.status
                            return final_status
                            # break

                    except Exception as e:
                        self.context.node.get_logger().info(e)
                        self.context.node.get_logger().info("Received (ERROR: '{}')".format(
                            reply.err.payload.to_string()))
                        continue

                time.sleep(1)
        except (KeyboardInterrupt):
            self.context.node.get_logger().info(f'Result: KeyboardInterrupt')
            pass

        finally:
            self.context.node.get_logger().info(f'Result: finally')
            feedback_sub.undeclare()
            session.close()
        return final_status

    def start_navigation(self, x: float, y: float):
        self.context.node.get_logger().info(f"Navigating to ({x}, {y}) synchronously...")
        self.navigation_started = True

        status = self.run_navigate_to_pose_client(x, y, namespace=self.namespace)

        if status == GoalStatus.STATUS_SUCCEEDED.value:
            self.context.node.get_logger().info(f"Nav status : {status}")
            self.context.node.get_logger().info(f"Successfully reached ({x}, {y})")
            self.state = RobotActionState.IN_PROGRESS
        else:
            self.context.node.get_logger().info(f"Nav status : {status}")
            self.context.node.get_logger().error(f"Navigation to ({x}, {y}) failed or aborted.")
            self.state = RobotActionState.FAILED

        self.result_received = True
        self.navigation_started = False

    def cleanup_navigation(self):
        self.context.node.get_logger().info("Cleaning up navigation resources.")
        self.navigation_started = False
        self.result_received = False

    def update_action(self) -> RobotActionState:
        if self.state in (RobotActionState.COMPLETED, RobotActionState.FAILED):
            self.cleanup_navigation()
            return self.state

        if not self.clean_path:
            self.state = RobotActionState.FAILED
            return self.state

        if self.current_index < len(self.clean_path):
            x, y = self.clean_path[self.current_index]
            self.start_navigation(x, y)

            if self.result_received:
                self.result_received = False
                self.current_index += 1

        if self.current_index >= len(self.clean_path):
            self.state = RobotActionState.COMPLETED
            self.cleanup_navigation()
            self.context.node.get_logger().info("All path points completed.")

        return self.state


class ActionFactory(RobotActionFactory):
    def __init__(self, context: RobotActionContext):
        super().__init__(context)
        self.supported_actions = ['clean']

    def supports_action(self, category: str) -> bool:
        return category in self.supported_actions

    def perform_action(self, category: str, description: dict, execution) -> RobotAction:
        if category == 'clean':
            return Clean(description, execution, self.context)
