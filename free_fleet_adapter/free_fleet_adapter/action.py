#!/usr/bin/env python3

# Copyright 2025 Open Source Robotics Foundation, Inc.
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
from typing import Annotated, Callable

from free_fleet_adapter.ros2_action_client import (
    Ros2ActionClient,
    Ros2ActionCodec,
)
import rclpy.node as Node
from rmf_adapter import RobotUpdateHandle
import rmf_adapter.easy_full_control as rmf_easy


class RobotActionState(StrEnum):
    IN_PROGRESS = 'in progress'
    CANCELING = 'canceling'
    CANCELED = 'canceled'
    COMPLETED = 'completed'
    FAILED = 'failed'


class RobotActionContext:
    def __init__(
        self,
        node: Node,
        robot_name: str,
        update_handle: RobotUpdateHandle,
        fleet_config: rmf_easy.FleetConfiguration,
        action_config: dict,
        get_battery_soc: Callable[[], float],
        get_map_name: Callable[[], str],
        get_pose: Callable[[], Annotated[list[float], 3] | None],
        create_ros2_action_client:
            Callable[[str, Ros2ActionCodec], Ros2ActionClient] | None = None
    ):
        self.node = node
        self.robot_name = robot_name
        self.update_handle = update_handle
        self.fleet_config = fleet_config
        self.action_config = action_config
        self.get_battery_soc = get_battery_soc
        self.get_map_name = get_map_name
        self.get_pose = get_pose
        self._create_ros2_action_client = create_ros2_action_client

    def create_ros2_action_client(
        self,
        action_name: str,
        codec: Ros2ActionCodec
    ) -> Ros2ActionClient:
        """
        Return a client that commands this robot's action_name endpoint.

        The action may be one the fleet adapter has never heard of: the
        endpoint name and the wire codecs come from the plugin, and the adapter
        only relays what the client reports back to RMF.
        """
        if self._create_ros2_action_client is None:
            raise NotImplementedError(
                f'The robot adapter of [{self.robot_name}] cannot command ROS '
                '2 actions, so no client is available for '
                f'[{action_name}].'
            )
        return self._create_ros2_action_client(action_name, codec)


class RobotAction(ABC):

    def __init__(self, context: RobotActionContext, execution):
        self.context = context
        self.execution = execution
        self.action_task_id = \
            self.context.update_handle.more().current_task_id()

    """
    This method is called on every update by the robot adapter to monitor the
    progress and completion of the action.
    Returns the state of the action.
    """
    @abstractmethod
    def update_action(self) -> RobotActionState:
        # To be populated in the plugins
        ...

    """
    This method is called when RMF no longer wants this action, either because
    the activity was stopped or because RMF took the execution away. The
    action's goals are still running on the robot at that point, and nothing
    else knows which goals those are, so this is the only chance to stop them.

    It must not wait for the robot to stop. Requesting the stop is all this
    does; the action keeps being updated until it reports a terminal state,
    and reporting that the robot has stopped is update_action's job. Note that
    a cancellation on an action endpoint is itself a service call, so an
    implementation that makes one blocks its caller for as long as that call
    takes.
    """
    def stop_action(self):
        # May be populated in the plugins
        ...

    """
    This method may be used by the action's implementation to cancel the current
    ongoing task.
    """
    def cancel_task_of_action(
        self,
        cancel_success: Callable[[], None],
        cancel_fail: Callable[[], None],
        label: str = ''
    ):
        self.context.node.get_logger().info(
            f'[{self.context.robot_name}] Cancel task requested for '
            f'[{self.action_task_id}]')

        def _on_cancel(result: bool):
            if result:
                self.context.node.get_logger().info(
                    f'[{self.context.robot_name}] Found task '
                    f'[{self.action_task_id}], cancelling...')
                cancel_success()
            else:
                self.context.node.get_logger().info(
                    f'[{self.context.robot_name}] Failed to cancel task '
                    f'[{self.action_task_id}]')
                cancel_fail()
        self.context.update_handle.more().cancel_task(
            self.action_task_id, [label], lambda result: _on_cancel(result))


class RobotActionFactory(ABC):

    def __init__(self, context: RobotActionContext):
        self.context = context

        if 'actions' not in context.action_config:
            raise KeyError(
                'List of supported actions is not provided in the action '
                'config! Unable to instantiate an ActionFactory.')
        self.actions = context.action_config['actions']

    """
    This method can be used to verify whether this ActionFactory supports
    the configured action.
    """
    @abstractmethod
    def supports_action(self, category: str) -> bool:
        # To be populated in the plugins
        ...

    """
    This method creates a Action object for the robot adapter to begin and
    interact with an action.
    """
    @abstractmethod
    def perform_action(
        self,
        category: str,
        description: dict,
        execution: rmf_easy.CommandExecution,
    ) -> RobotAction:
        # To be populated in the plugins
        ...
