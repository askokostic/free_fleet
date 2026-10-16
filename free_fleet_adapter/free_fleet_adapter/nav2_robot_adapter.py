#!/usr/bin/env python3

# Copyright 2024 Open Source Robotics Foundation, Inc.
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
from dataclasses import dataclass
import importlib
from threading import Lock, RLock
import time
from typing import Annotated, Callable

from free_fleet.convert import transform_stamped_to_ros2_msg
from free_fleet.ros2_types import (
    GeometryMsgs_PoseStamped,
    Header,
    SensorMsgs_BatteryState,
    StdMsgs_Empty,
    StdSrvs_SetBool_Request,
    StdSrvs_SetBool_Response,
    TFMessage,
    Time,
)
from free_fleet.utils import (
    namespacify,
)
from free_fleet_adapter.action import (
    RobotActionContext,
    RobotActionState,
)
from free_fleet_adapter.nav2_action_codecs import (
    DOCK_ROBOT_ACTION_NAME,
    DockRobotCodec,
    DockRobotGoal,
    NAVIGATE_TO_POSE_ACTION_NAME,
    NavigateToPoseCodec,
    NavigateToPoseGoal,
)
from free_fleet_adapter.robot_adapter import (
    ExecutionHandle,
    RobotAdapter
)
from free_fleet_adapter.ros2_action_client import (
    GoalNotAccepted,
    Ros2ActionClient,
    Ros2ActionCodec,
    Ros2GoalState,
)

from geometry_msgs.msg import TransformStamped
from pycdr2 import IdlStruct
import rclpy
import rmf_adapter.easy_full_control as rmf_easy
from rmf_adapter.robot_update_handle import (
    ActionExecution,
    ActivityIdentifier,
    Tier
)
from std_srvs.srv import SetBool
from tf_transformations import euler_from_quaternion
import zenoh


# CDR type matching nav2_evacuation_monitor/msg/EvacuationStatus of
# nav2_zone_evacuation
@dataclass
class EvacuationStatus(IdlStruct):
    header: Header
    state: str
    inside_zone: bool
    evacuation_goal_in_progress: bool
    active_zone_name: str
    goal_pose: GeometryMsgs_PoseStamped
    next_retry_time: Time
    detail: str


class Nav2TfHandler:

    def __init__(self, robot_name, zenoh_session, tf_buffer, node,
                 robot_frame='base_footprint', map_frame='map'):
        self.robot_name = robot_name
        self.zenoh_session = zenoh_session
        self.node = node
        self.tf_buffer = tf_buffer
        self.robot_frame = robot_frame
        self.map_frame = map_frame

        def _tf_callback(sample: zenoh.Sample, static: bool = False):
            try:
                transform = TFMessage.deserialize(sample.payload.to_bytes())
            except Exception as e:
                self.node.get_logger().debug(
                    f'Failed to deserialize TF payload: {type(e)}: {e}'
                )
                return None
            set_transform = (self.tf_buffer.set_transform_static if static
                             else self.tf_buffer.set_transform)
            for zt in transform.transforms:
                t = transform_stamped_to_ros2_msg(zt)
                t.header.frame_id = namespacify(zt.header.frame_id,
                                                self.robot_name)
                t.child_frame_id = namespacify(zt.child_frame_id,
                                               self.robot_name)
                set_transform(t, f'{self.robot_name}_TfListener')

        def _tf_static_callback(sample: zenoh.Sample):
            _tf_callback(sample, static=True)

        self.tf_sub = self.zenoh_session.declare_subscriber(
            namespacify('tf', self.robot_name),
            _tf_callback
        )

        self.tf_static_sub = self.zenoh_session.declare_subscriber(
            namespacify('tf_static', self.robot_name),
            _tf_static_callback
        )

    def get_transform(self) -> TransformStamped | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                namespacify(self.map_frame, self.robot_name),
                namespacify(self.robot_frame, self.robot_name),
                rclpy.time.Time()
                )
            return transform
        except Exception as err:
            self.node.get_logger().info(
                f'Unable to get transform between {self.robot_frame} '
                f'and {self.map_frame}: {type(err)}: {err}'
            )
        return None


class RobotClaim(ABC):
    """
    The one thing commanding a robot, and what it takes to get it back.

    Whatever is driving the robot holds a claim on it, and holds it across
    every goal it sends. A plugin issuing several goals in sequence keeps its
    claim through the gaps between them, so nothing can slip a command into
    one and drive the robot somewhere the plugin never asked for.

    Asking a claim to stop does not release it. What releases it is the
    report that the robot stopped, because it is the robot being handed over
    and not the request - or, if that report never comes, the expiry of the
    bounded wait in Nav2RobotAdapter._admit_pending_command, which no operator
    is left to find out about on their own.
    """

    def __init__(self, subject: str, exec_handle: ExecutionHandle):
        self.subject = subject
        self.exec_handle = exec_handle
        self.stop_requested_sec: float | None = None
        self.stop_timeout_reported = False
        # The evacuation monitor cancels all goals when it closes the gate.
        # The client can report that cancel only after the evacuation.
        self.gate_closed_by_monitor = False

    @property
    def activity(self) -> ActivityIdentifier | None:
        return self.exec_handle.activity

    @property
    def stop_requested(self) -> bool:
        return self.stop_requested_sec is not None

    """
    This method asks whatever is driving the robot to stop. Requesting is all
    it does: the claim is released by the report that the robot stopped, which
    arrives on a later update.
    """
    @abstractmethod
    def request_stop(self):
        ...


class NavigationClaim(RobotClaim):
    """A navigation command RMF gave the robot, and the goal carrying it."""

    def __init__(
        self,
        subject: str,
        client: Ros2ActionClient,
        exec_handle: ExecutionHandle
    ):
        super().__init__(subject, exec_handle)
        self.client = client
        # A node on the robot canceled or ended the goal before this stop
        self.goal_over_before_stop = False

    def request_stop(self):
        with self.exec_handle.mutex:
            self.goal_over_before_stop = self.client.cancel()

    def finish(self):
        """Report to RMF that the command this goal was carrying is over."""
        execution = self.exec_handle.execution
        if execution is not None:
            execution.finished()
            self.exec_handle.execution = None


class ActionClaim(RobotClaim):
    """
    A plugin action performing an RMF activity on the robot.

    The action is what knows which goals are driving the robot, so a claim
    always has one: execute_action refuses a category it has no plugin for
    before it asks for the robot at all.
    """

    def request_stop(self):
        self.exec_handle.action.stop_action()


class PendingCommand:
    """A command waiting for the robot to be let go by whatever has it."""

    def __init__(
        self, subject: str, start: Callable[[], None], navigation: bool
    ):
        self.subject = subject
        self.start = start
        # RMF can plan a navigation again, but not an action
        self.navigation = navigation


class Nav2RobotAdapter(RobotAdapter):
    MAX_REPLAN_COUNTS = 5
    DEFAULT_SERVICE_CALL_TIMEOUT_SEC = 1.0
    # How long an incoming command waits for the report that the robot
    # stopped before it is admitted without one, and an operator is told.
    DEFAULT_STOP_TIMEOUT_SEC = 10.0
    # The gate is closed when the last evacuation monitor status is older
    EVACUATION_STATUS_MAX_AGE_SEC = 3.0
    # An issue is raised if no evacuation monitor status came in this time
    EVACUATION_STATUS_FIRST_WAIT_SEC = 10.0
    # After a goal that another node canceled, the time in which an
    # evacuation monitor status can show that the monitor canceled it
    CANCEL_DECISION_WINDOW_SEC = 2.0

    def __init__(
        self,
        name: str,
        configuration,
        robot_config_yaml,
        plugin_config: dict | None,
        node,
        zenoh_session,
        fleet_handle,
        fleet_config: rmf_easy.FleetConfiguration | None,
        tf_buffer
    ):
        RobotAdapter.__init__(self, name, node, fleet_handle)

        self.configuration = configuration
        self.robot_config_yaml = robot_config_yaml
        self.zenoh_session = zenoh_session
        self.fleet_config = fleet_config
        self.tf_buffer = tf_buffer

        # the one command that has the robot, and the one waiting for it
        self.claim: RobotClaim | None = None
        self.pending_command: PendingCommand | None = None
        # Every change of hands is made under this. The robot is handed over
        # on the update thread, and it is RMF's own thread that asks for it.
        self.claim_mutex = RLock()

        self.map_name = self.robot_config_yaml['initial_map']
        default_map_frame = 'map'
        default_robot_frame = 'base_footprint'
        self.map_frame = self.robot_config_yaml.get('map_frame', default_map_frame)
        self.robot_frame = self.robot_config_yaml.get('robot_frame', default_robot_frame)
        self.service_call_timeout_sec = self.robot_config_yaml.get(
            'service_call_timeout_sec', self.DEFAULT_SERVICE_CALL_TIMEOUT_SEC)
        self.stop_timeout_sec = self.robot_config_yaml.get(
            'stop_timeout_sec', self.DEFAULT_STOP_TIMEOUT_SEC)
        # Hold the goals during an evacuation of nav2_evacuation_monitor
        self.use_evacuation_monitor = self.robot_config_yaml.get(
            'use_evacuation_monitor', False)

        # TODO(ac): Only use full battery if sim is indicated
        self.battery_soc = 1.0

        self.replan_counts = 0
        self.nav_issue_ticket = None
        # Every claim that would not report a stop in time, with the issue
        # raised about it. Both halves are kept: dropping the ticket retracts
        # the issue from the operator's view, and releasing the execution
        # fires the same completion trigger that finishing it would.
        self.stop_timeout_issues = []

        # Maps action name to plugin name
        self.action_to_plugin_name = {}
        # Maps plugin name to action factory
        self.action_factories = {}

        # Maps a ROS 2 action endpoint name to the client that drives it
        self.action_clients: dict[str, Ros2ActionClient] = {}

        self.tf_handler = Nav2TfHandler(
            self.name,
            self.zenoh_session,
            self.tf_buffer,
            self.node,
            robot_frame=self.robot_frame,
            map_frame=self.map_frame
        )

        def _battery_state_callback(sample: zenoh.Sample):
            battery_state = SensorMsgs_BatteryState.deserialize(
                sample.payload.to_bytes()
            )
            self.battery_soc = battery_state.percentage

        self.battery_state_sub = self.zenoh_session.declare_subscriber(
            namespacify('battery_state', name),
            _battery_state_callback
        )

        self._start_heartbeat()

        # The gate holds the goals of the adapter and of the plugins. See
        # _gate_reason.
        self.gate_mutex = Lock()
        self.evacuation_status: EvacuationStatus | None = None
        self.evacuation_status_sec: float | None = None
        # The monitor had its data (pose, zones) one time. After that,
        # WAITING_FOR_DATA can hide a robot in a zone.
        self.evacuation_monitor_had_data = False
        self.start_sec = self._monotonic_sec()
        self.cancel_window_end_sec: float | None = None
        self.remote_hold = False
        # Used only on the update thread
        self.gate_closed = False
        self.no_status_issue = None
        self.monitor_issue = None
        self.monitor_issue_state = None
        # RMF waits for a navigation command that the gate stopped
        self.replan_on_gate_open = False

        def _evacuation_status_callback(sample: zenoh.Sample):
            # Only the receive time is used: the robot can use sim time
            try:
                status = EvacuationStatus.deserialize(
                    sample.payload.to_bytes())
            except Exception as e:
                self.node.get_logger().debug(
                    f'Failed to deserialize evacuation status: {type(e)}: {e}'
                )
                return
            with self.gate_mutex:
                self.evacuation_status = status
                self.evacuation_status_sec = self._monotonic_sec()
                if status.state != 'WAITING_FOR_DATA':
                    self.evacuation_monitor_had_data = True

        if self.use_evacuation_monitor:
            self.evacuation_status_sub = self.zenoh_session.declare_subscriber(
                namespacify('evacuation_monitor/status', name),
                _evacuation_status_callback
            )

        # Initialize robot
        init_timeout_sec = self.robot_config_yaml.get('init_timeout_sec', 10)
        self.node.get_logger().info(f'Initializing robot [{self.name}]...')
        init_robot_pose = rclpy.Future()

        def _get_init_pose():
            robot_pose = self.get_pose()
            if robot_pose is not None:
                init_robot_pose.set_result(robot_pose)
                init_robot_pose.done()

        init_pose_timer = self.node.create_timer(1, _get_init_pose)
        rclpy.spin_until_future_complete(
            self.node, init_robot_pose, timeout_sec=init_timeout_sec
        )

        if init_robot_pose.result() is None:
            error_message = \
                f'Timeout trying to initialize robot [{self.name}]'
            self.node.get_logger().error(error_message)
            raise RuntimeError(error_message)

        self.node.destroy_timer(init_pose_timer)
        state = rmf_easy.RobotState(
            self.get_map_name(),
            init_robot_pose.result(),
            self.get_battery_soc()
        )

        if self.fleet_handle is None:
            self.node.get_logger().warn(
                f'Fleet unavailable, skipping adding robot [{self.name}] '
                'to fleet.'
            )
            return

        self.update_handle = self.fleet_handle.add_robot(
            self.name,
            state,
            self.configuration,
            rmf_easy.RobotCallbacks(
                lambda destination, execution: self.navigate(
                    destination, execution
                ),
                lambda activity: self.stop(activity),
                lambda category, description, execution: self.execute_action(
                    category, description, execution
                )
            )
        )
        if not self.update_handle:
            error_message = \
                f'Failed to add robot [{self.name}] to fleet ' \
                f'[{self.fleet_handle.more().fleet_name()}], this is most ' \
                'likely due to a configuration error.'
            self.node.get_logger().error(error_message)
            raise RuntimeError(error_message)

        if self.use_evacuation_monitor:
            self.hold_service = self.node.create_service(
                SetBool, f'{self.name}/hold', self._hold_callback)

        if self.fleet_config is None:
            self.node.get_logger().info(
                'No fleet configuration provided for RobotAdapter of '
                f'[{self.name}]. No plugin actions will be loaded.'
            )
            return

        # Import and store plugin actions and action factories
        if plugin_config is not None:
            for plugin_name, action_config in plugin_config.items():
                try:
                    module = action_config['module']
                    plugin = importlib.import_module(module)
                    action_context = RobotActionContext(
                        self.node,
                        self.name,
                        self.update_handle,
                        self.fleet_config,
                        action_config,
                        self.get_battery_soc,
                        self.get_map_name,
                        self.get_pose,
                        self.create_ros2_action_client
                    )
                    action_factory = plugin.ActionFactory(action_context)
                    for action in action_factory.actions:
                        # Verify that this action is not duplicated across plugins
                        target_plugin = self.action_to_plugin_name.get(action)
                        if (target_plugin is not None and
                                target_plugin != plugin_name):
                            raise Exception(
                                f'Action [{action}] is indicated to be supported '
                                f'by multiple plugins: {target_plugin} and '
                                f'{plugin_name}. The fleet adapter is unable to '
                                f'select the intended plugin to be paired for '
                                f'this action. Please ensure that action names '
                                f'are not duplicated across supported plugins. '
                                f'Unable to create ActionFactory for '
                                f'{plugin_name}. Robot [{self.name}] will not be '
                                f'able to perform actions associated with this '
                                f'plugin.'
                            )
                        # Verify that this ActionFactory supports this action
                        if not action_factory.supports_action(action):
                            raise ValueError(
                                f'The plugin config provided [{action}] as a '
                                f'performable action, but it is not a supported '
                                f'action in the {plugin_name} ActionFactory!'
                            )
                        self.action_to_plugin_name[action] = plugin_name
                    self.action_factories[plugin_name] = action_factory
                except KeyError:
                    self.node.get_logger().info(
                        f'Unable to create ActionFactory for {plugin_name}! '
                        f'Configured plugin config is invalid. '
                        f'Robot [{self.name}] will not be able to perform '
                        f'actions associated with this plugin.'
                    )
                except ImportError:
                    self.node.get_logger().info(
                        f'Unable to import module for {plugin_name}! '
                        f'Robot [{self.name}] will not be able to perform '
                        f'actions associated with this plugin.'
                    )

    def create_ros2_action_client(
        self,
        action_name: str,
        codec: Ros2ActionCodec
    ) -> Ros2ActionClient:
        """
        Return this robot's client for the action_name endpoint.

        The adapter needs to know nothing about the action to hand out a client
        for it: the endpoint name and the wire codecs come from the caller. One
        client is kept per endpoint and reused across goals, so each endpoint
        declares its feedback subscription exactly once.
        """
        client = self.action_clients.get(action_name)
        if client is not None:
            if type(client.codec) is not type(codec):
                raise ValueError(
                    f'Endpoint [{action_name}] of robot [{self.name}] is '
                    f'already driven with codec '
                    f'[{type(client.codec).__name__}], so it cannot also be '
                    f'driven with codec [{type(codec).__name__}].'
                )
            return client

        client = Ros2ActionClient(
            self.node,
            self.zenoh_session,
            self.name,
            action_name,
            codec,
            self.service_call_timeout_sec,
            self._gate_reason
        )
        self.action_clients[action_name] = client
        return client

    def get_battery_soc(self) -> float:
        return self.battery_soc

    def get_map_name(self) -> str:
        return self.map_name

    def get_pose(self) -> Annotated[list[float], 3] | None:
        transform = self.tf_handler.get_transform()
        if transform is None:
            error_message = \
                f'Failed to update robot [{self.name}]: Unable to get ' \
                f'transform between {self.robot_frame} and {self.map_frame}'
            self.node.get_logger().info(error_message)
            return None

        orientation = euler_from_quaternion([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ])
        robot_pose = [
            transform.transform.translation.x,
            transform.transform.translation.y,
            orientation[2]
        ]
        return robot_pose

    # TODO(ac): issue ticket can be more generic for execute actions too
    def create_nav_issue_ticket(self, category, msg, nav_goal_id=None):
        tier = Tier.Error
        detail = {
            'nav_goal_id': f'{nav_goal_id}',
            'message': msg
        }
        nav_issue_ticket = \
            self.update_handle.more().create_issue(tier, category, detail)
        self.node.get_logger().info(
            f'Created [{category}] issue ticket for robot [{self.name}] with '
            f'nav goal ID [{nav_goal_id}]')
        return nav_issue_ticket

    def update(self, state: rmf_easy.RobotState):
        if self.update_handle is None:
            error_message = \
                f'Failed to update robot {self.name}, robot adapter has not ' \
                'yet been initialized with a fleet update handle.'
            self.node.get_logger().error(error_message)
            return

        claim = self.claim
        match claim:
            case NavigationClaim():
                self._report_navigation_outcome(claim)
            case ActionClaim():
                self._report_action_outcome(claim)

        if self.use_evacuation_monitor:
            self._update_gate()
        self._admit_pending_command()

        claim = self.claim
        activity_identifier = None if claim is None else claim.activity
        self.update_handle.update(state, activity_identifier)

    def _report_action_outcome(self, claim: ActionClaim):
        """
        Carry the running action forward, and tell RMF what became of it.

        An execution RMF has withdrawn is no longer the action's to finish:
        RMF has already given the event its own terminal status, and finishing
        it here would stamp Completed over that. The action is told instead,
        and keeps the robot until it reports that the robot has stopped.

        As after a navigation goal that another node canceled, a
        CANCELED action that the adapter did not ask to stop keeps the gate
        closed until the monitor status shows if the monitor canceled it.
        """
        execution = claim.exec_handle.execution
        if not execution.okay():
            self._request_stop(claim)

        current_action_state = claim.exec_handle.action.update_action()
        match current_action_state:
            case RobotActionState.CANCELED | \
                    RobotActionState.COMPLETED | \
                    RobotActionState.FAILED:
                self.node.get_logger().info(
                    f'Robot [{self.name}] current action '
                    f'[{current_action_state}]'
                )
                if current_action_state is not RobotActionState.CANCELED:
                    execution.finished()
                self._release_claim(claim)
                unrequested_cancel = (
                    current_action_state is RobotActionState.CANCELED and
                    not claim.stop_requested)
                if unrequested_cancel and not claim.gate_closed_by_monitor:
                    self._start_cancel_decision_window()

    def _report_navigation_outcome(self, claim: NavigationClaim):
        """
        Tell RMF what became of the navigation goal this command is carrying.

        The client reports the goal. Deciding what that means for RMF, be it
        ending the command, raising an issue ticket or replanning, is the
        adapter's job, and stays here whatever action is being driven.

        A client is shared by every goal sent to its endpoint, so what it
        reports is only this command's business while the two still name the
        same goal. Until the goal has been sent, and once the client has moved
        on to another one, there is nothing here to report.
        """
        goal_id = claim.exec_handle.goal_id
        if goal_id is None:
            return

        outcome = claim.client.outcome()
        if outcome.goal_id != goal_id:
            return

        if outcome.state is Ros2GoalState.EXECUTING:
            return

        if outcome.state is Ros2GoalState.UNDECODABLE:
            # The goal is still running; only our reading of it failed
            self.node.get_logger().warn(
                f'Robot [{self.name}] could not read the outcome of its '
                f'navigation goal: {outcome.detail}'
            )
            return

        canceled_on_robot = (
            self.use_evacuation_monitor and
            outcome.state is Ros2GoalState.CANCELED and
            claim.goal_over_before_stop)
        if claim.stop_requested and not canceled_on_robot:
            # RMF is not waiting on a command it has already taken back, and
            # finishing one would step a path RMF has replaced. All this goal
            # is still good for is the one thing being waited on: that it has
            # stopped driving the robot.
            self._release_claim(claim)
            return

        match outcome.state:
            case Ros2GoalState.ABORTED:
                self._release_claim(claim)
                self.nav_issue_ticket = self.create_nav_issue_ticket(
                    'navigation',
                    outcome.detail,
                    outcome.goal_id
                )
                self._trigger_replan()

            case Ros2GoalState.SUCCEEDED:
                self._release_claim(claim)
                self.replan_counts = 0
                claim.finish()
                if self.nav_issue_ticket is not None:
                    self.nav_issue_ticket.resolve({})
                    self.nav_issue_ticket = None
                    self.node.get_logger().info(
                        'Navigation issue ticket has been resolved'
                    )

            case Ros2GoalState.CANCELED:
                # The adapter did not ask for this cancel, or the goal was
                # canceled before the adapter asked. Thus another node
                # canceled the goal and the robot did not arrive. RMF plans
                # the command again, as after an abort. With the evacuation
                # monitor, the replan waits for the gate: if the monitor did
                # not close the gate while the goal ran, the gate stays
                # closed until the monitor status shows if the monitor
                # canceled it.
                self._release_claim(claim)
                self.node.get_logger().warn(
                    f'Robot [{self.name}]: another node canceled navigation '
                    f'goal {outcome.goal_id}'
                )
                if not self.use_evacuation_monitor:
                    self._trigger_replan()
                    return
                self.replan_counts = 0
                self.replan_on_gate_open = True
                if not claim.gate_closed_by_monitor:
                    self._start_cancel_decision_window()

            case _:
                # No goal is running under this ID, so there is nothing this
                # command can be told about it
                self.node.get_logger().warn(
                    f'Robot [{self.name}] has no navigation goal '
                    f'{outcome.goal_id} to report on'
                )

    def _start_cancel_decision_window(self):
        """Close the gate until the monitor status shows who canceled."""
        if not self.use_evacuation_monitor:
            return
        with self.gate_mutex:
            self.cancel_window_end_sec = (
                self._monotonic_sec() + self.CANCEL_DECISION_WINDOW_SEC)

    def navigate(
        self,
        destination: rmf_easy.Destination,
        execution: rmf_easy.CommandExecution
    ):
        subject = (
            f'a navigation to {destination.position} on map '
            f'[{destination.map}]'
        )
        self._take_command(
            subject,
            lambda: self._start_navigation(subject, destination, execution),
            navigation=True
        )

    def _start_navigation(
        self,
        subject: str,
        destination: rmf_easy.Destination,
        execution: rmf_easy.CommandExecution
    ):
        """Send the goal that carries out a navigation, and claim the robot."""
        if destination.map != self.map_name:
            # TODO(ac): test this map related replanning behavior
            self.node.get_logger().error(
                f'Destination is on map [{destination.map}], while robot '
                f'[{self.name}] is on map [{self.map_name}]'
            )
            self._trigger_replan()
            return

        if destination.dock is not None:
            self.node.get_logger().info(
                f'Commanding [{self.name}] to navigate to {destination.dock} '
                f'on map [{destination.map}]'
            )

            client = self.create_ros2_action_client(
                DOCK_ROBOT_ACTION_NAME, DockRobotCodec())
            goal = DockRobotGoal(dock_id=destination.dock)
        else:
            self.node.get_logger().info(
                f'Commanding [{self.name}] to navigate to {destination.position} '
                f'on map [{destination.map}]'
            )

            time_now = self.node.get_clock().now().seconds_nanoseconds()
            client = self.create_ros2_action_client(
                NAVIGATE_TO_POSE_ACTION_NAME, NavigateToPoseCodec())
            goal = NavigateToPoseGoal(
                frame_id=self.map_frame,
                x=destination.position[0],
                y=destination.position[1],
                z=0.0,
                yaw=destination.position[2],
                stamp=Time(sec=time_now[0], nanosec=time_now[1])
            )

        claim = NavigationClaim(subject, client, ExecutionHandle(execution))
        self.claim = claim

        try:
            claim.exec_handle.set_goal_id(client.start(goal))
        except GoalNotAccepted as e:
            self.node.get_logger().error(
                f'Robot [{self.name}] could not start its navigation goal: {e}'
            )
            self.claim = None
            # The goal never took command, so its mutex is still held here;
            # cancel without it, in case the robot is driving the goal anyway
            client.cancel()
            self._trigger_replan()

    def stop(self, activity: ActivityIdentifier):
        # Whatever has the robot keeps it until it reports that it stopped.
        # Dropping it here would leave the goal it sent running, with nobody
        # left who knows which goal that is.
        claim = self.claim
        if claim is not None:
            self._request_stop(claim)

    def execute_action(
        self,
        category: str,
        description: dict,
        execution: ActionExecution,
    ):
        action_factory = self._action_factory_for(category)
        if action_factory is None:
            # TODO(ac): change map using map_server load_map, and set initial
            # position again with /initialpose
            # TODO(ac): docking
            # We should never reach this point after initialization.
            # Refused before the robot is asked for, so whatever has it is
            # left undisturbed.
            error_message = \
                f'RobotAction [{category}] was not configured for this fleet.'
            self.node.get_logger().error(error_message)
            raise NotImplementedError(error_message)

        subject = f'an action [{category}]'
        self._take_command(
            subject,
            lambda: self._start_action(
                subject, action_factory, category, description, execution),
            navigation=False
        )

    def _action_factory_for(self, category: str):
        """Find the plugin that answers to this category, if any does."""
        plugin_name = self.action_to_plugin_name.get(category)
        if plugin_name:
            return self.action_factories.get(plugin_name)

        for factory in self.action_factories.values():
            if factory.supports_action(category):
                factory.actions.append(category)
                return factory
        return None

    def _start_action(
        self,
        subject: str,
        action_factory,
        category: str,
        description: dict,
        execution: ActionExecution,
    ):
        """Begin a plugin action, and claim the robot for it."""
        robot_action = action_factory.perform_action(
            category, description, execution
        )
        exec_handle = ExecutionHandle(execution)
        exec_handle.set_action(robot_action)
        self.claim = ActionClaim(subject, exec_handle)

    def _take_command(
        self, subject: str, start: Callable[[], None], navigation: bool
    ):
        """
        Give the robot to a new command, once whatever has it lets go.

        A robot answers to one commander at a time, so an incoming command
        does not simply take it from the one holding it: the stop is
        requested here, and the command waits to be admitted until the robot
        is reported stopped - or until that wait runs out, in
        _admit_pending_command.
        """
        gate_reason = self._gate_reason()
        with self.claim_mutex:
            claim = self.claim
            if claim is None and gate_reason is None:
                self.pending_command = None
                start()
                return

            superseded = self.pending_command
            if superseded is not None:
                self.node.get_logger().warn(
                    f'Robot [{self.name}] was given {subject} while '
                    f'{superseded.subject} was still waiting for the robot; '
                    'only the newer one will be carried out'
                )
            self.pending_command = PendingCommand(subject, start, navigation)
            if claim is None:
                self.node.get_logger().info(
                    f'Robot [{self.name}] was given {subject}. It waits, '
                    f'because {gate_reason}'
                )
                return
            self.node.get_logger().info(
                f'Robot [{self.name}] was given {subject} while '
                f'{claim.subject} still has it, so that is being asked to stop'
            )

        self._request_stop(claim)

    def _request_stop(self, claim: RobotClaim):
        """
        Tell the claim holder, once, that the robot is wanted back.

        The clock on the wait for the report that the robot stopped starts
        here, so a holder that never reports cannot keep the robot forever.

        The cancel is made under the same lock that decides who holds the
        robot, and not after it, so that a cancel never outlives its claim and
        cancels a goal that the claim no longer owns.
        """
        with self.claim_mutex:
            if claim.stop_requested or self.claim is not claim:
                return
            claim.stop_requested_sec = self._now_sec()
            claim.request_stop()

    def _release_claim(self, claim: RobotClaim):
        """Let the robot go, now that what was driving it has stopped."""
        with self.claim_mutex:
            if self.claim is claim:
                self.claim = None

    def _admit_pending_command(self):
        """
        Hand the robot to the command waiting for it, and mind the clock.

        A command waits for the report that the robot stopped, but not
        forever: a holder that never reports would strand it, and every
        command after it. Once the wait has run out the command is admitted
        without that report and an operator is told, because the robot may
        still be moving under something nothing is watching any more.

        A holder that will not report is worth saying so about even when
        nothing is waiting for the robot - after a kill, nothing is. There is
        no handover to make there, so the robot stays held; what an operator
        gets is the same issue, raised as soon as the wait runs out rather
        than whenever the next command happens to arrive.
        """
        with self.claim_mutex:
            claim = self.claim
            if claim is not None and not self._stop_wait_expired(claim):
                return

            pending = self.pending_command
            if pending is None:
                if claim is not None:
                    self._raise_stop_timeout_issue(claim, admitted=None)
                return

            if self._gate_reason() is not None:
                if claim is not None:
                    self._raise_stop_timeout_issue(claim, admitted=None)
                return

            self.pending_command = None
            if claim is not None:
                self._raise_stop_timeout_issue(claim, admitted=pending)
                self.claim = None
            pending.start()

    def _stop_wait_expired(self, claim: RobotClaim) -> bool:
        """Say whether this claim has run out of time to report a stop."""
        if claim.stop_requested_sec is None:
            return False
        waited = self._now_sec() - claim.stop_requested_sec
        return waited >= self.stop_timeout_sec

    def _raise_stop_timeout_issue(
        self,
        claim: RobotClaim,
        admitted: PendingCommand | None
    ):
        """Tell an operator that a holder would not report a stop in time."""
        if claim.stop_timeout_reported:
            return
        claim.stop_timeout_reported = True

        message = (
            f'{claim.subject} holding robot [{self.name}] did not report that '
            f'the robot had stopped within {self.stop_timeout_sec}s, '
        )
        if admitted is None:
            message += (
                'and no command was admitted, so it is still held. '
                'The robot may still be moving under a command that has '
                'stopped answering.'
            )
        else:
            message += (
                f'so {admitted.subject} was admitted without it. The robot '
                'may still be moving under a command nothing is watching '
                'any more.'
            )
        self.node.get_logger().error(message)
        self.stop_timeout_issues.append((
            claim,
            self.update_handle.more().create_issue(
                Tier.Error,
                'robot_stop_timeout',
                {
                    'held_by': claim.subject,
                    'admitted': None if admitted is None else admitted.subject,
                    'waited_sec': self.stop_timeout_sec,
                    'message': message,
                }
            )
        ))

    def _gate_reason(self) -> str | None:
        """
        Return why the robot may get no new goal now, or None.

        The gate holds the goals of the adapter and of the plugins while the
        evacuation monitor on the robot moves the robot out of a zone, and
        while the robot is held. A goal that is running is not stopped.
        Without use_evacuation_monitor, the gate is always open.
        """
        if not self.use_evacuation_monitor:
            return None
        with self.gate_mutex:
            status = self.evacuation_status
            status_sec = self.evacuation_status_sec
            if self.remote_hold:
                return 'the robot is held (hold service)'
            if self.cancel_window_end_sec is not None:
                return 'another node canceled a goal'
            had_data = self.evacuation_monitor_had_data
        if status is None:
            return None
        if (self._monotonic_sec() - status_sec >
                self.EVACUATION_STATUS_MAX_AGE_SEC):
            return 'the evacuation monitor status is too old'
        if status.state == 'WAITING_FOR_DATA' and had_data:
            return 'the evacuation monitor has no robot pose or zone data'
        return self._monitor_reason(status)

    @staticmethod
    def _monitor_reason(status: EvacuationStatus) -> str | None:
        """Return why this evacuation monitor status closes the gate."""
        if status.state in ('HOLD', 'FAILED'):
            return f'the evacuation monitor state is {status.state}'
        if status.inside_zone:
            return f'the robot is in zone [{status.active_zone_name}]'
        if status.evacuation_goal_in_progress:
            return 'the evacuation exit goal is in progress'
        return None

    def _update_gate(self):
        """
        End the decision window of a cancel, and replan when the gate opens.

        RMF planned a pending navigation from where the robot was before the
        gate closed, so it is dropped and RMF plans again. A pending action
        is kept, because RMF does not plan it again.
        """
        now = self._monotonic_sec()
        with self.gate_mutex:
            status = self.evacuation_status
            status_sec = self.evacuation_status_sec
            fresh = status is not None and (
                now - self.evacuation_status_sec <=
                self.EVACUATION_STATUS_MAX_AGE_SEC)
            monitor_reason = self._monitor_reason(status) if fresh else None
            claim = self.claim
            if monitor_reason is not None and claim is not None:
                claim.gate_closed_by_monitor = True
            if self.cancel_window_end_sec is not None and (
                    monitor_reason is not None or
                    now >= self.cancel_window_end_sec):
                self.cancel_window_end_sec = None

        self._update_monitor_issues(status, status_sec, now)

        reason = self._gate_reason()
        if reason is not None and not self.gate_closed:
            self.node.get_logger().warn(
                f'Robot [{self.name}] gets no new goal, because {reason}')
        elif reason is None and self.gate_closed:
            self.node.get_logger().info(
                f'Robot [{self.name}] can get new goals again')
            with self.claim_mutex:
                pending = self.pending_command
                if pending is not None and pending.navigation:
                    self.pending_command = None
                    self.replan_on_gate_open = True
        if reason is None and self.replan_on_gate_open:
            # Not limited by MAX_REPLAN_COUNTS: without this replan, RMF
            # waits for the navigation for ever
            self.replan_on_gate_open = False
            self.replan_counts = 0
            self._trigger_replan()
        self.gate_closed = reason is not None

    def _update_monitor_issues(
        self, status: EvacuationStatus | None, status_sec, now
    ):
        """Keep one issue for a missing status, and one for HOLD or FAILED."""
        last_sec = self.start_sec if status is None else status_sec
        if (self.no_status_issue is None and now - last_sec >
                self.EVACUATION_STATUS_FIRST_WAIT_SEC):
            message = (
                f'No evacuation monitor status from robot [{self.name}] '
                f'in {self.EVACUATION_STATUS_FIRST_WAIT_SEC}s. Before the '
                'first status, RMF can send goals during an evacuation. '
                'After it, the robot gets no new goal.'
            )
            self.node.get_logger().error(message)
            self.no_status_issue = self.update_handle.more().create_issue(
                Tier.Error, 'evacuation_monitor_status_missing',
                {'message': message})
        elif (self.no_status_issue is not None and
                now - last_sec <= self.EVACUATION_STATUS_MAX_AGE_SEC):
            self.no_status_issue.resolve({})
            self.no_status_issue = None
        if status is None:
            return

        state = status.state if status.state in ('HOLD', 'FAILED') else None
        if self.monitor_issue is not None and state != self.monitor_issue_state:
            self.monitor_issue.resolve({})
            self.monitor_issue = None
        if state is not None and self.monitor_issue is None:
            self.monitor_issue_state = state
            self.monitor_issue = self.update_handle.more().create_issue(
                Tier.Error, 'evacuation_monitor',
                {'state': state, 'zone': status.active_zone_name,
                 'message': status.detail})

    def _hold_callback(self, request, response):
        """
        Hold or release the robot: the adapter gate and the monitor.

        A hold closes the gate and stops the goal of the adapter, also if the
        monitor does not answer. During a task action, it cancels the goals of
        the action, and the action fails, as after a cancel by the monitor. A
        release clears the hold and the decision window of the adapter, also
        if the monitor does not answer. Then a monitor that
        is still in HOLD, or a status that is too old, keeps the gate closed.
        """
        with self.gate_mutex:
            self.remote_hold = request.data
            if not request.data:
                self.cancel_window_end_sec = None
        claim = self.claim
        if request.data and isinstance(claim, NavigationClaim):
            if not claim.stop_requested:
                # RMF waits for this navigation
                self.replan_on_gate_open = True
            self._request_stop(claim)
        elif request.data and claim is not None:
            # stop_action() ends the action CANCELED, and RMF then waits for
            # ever. A canceled goal makes the action fail and end its task.
            for client in list(self.action_clients.values()):
                client.cancel()
        response.success, response.message = self._forward_hold(request.data)
        self.node.get_logger().info(
            f'Robot [{self.name}] hold {request.data}: '
            f'success {response.success}, {response.message}'
        )
        return response

    def _forward_hold(self, hold: bool) -> tuple[bool, str]:
        """Call the hold service of the evacuation monitor on the robot."""
        request = StdSrvs_SetBool_Request(data=hold)
        for reply in self.zenoh_session.get(
            namespacify('evacuation_monitor/hold', self.name),
            payload=request.serialize(),
            timeout=self.service_call_timeout_sec
        ):
            if reply.ok is None:
                return False, 'the evacuation monitor hold call failed'
            try:
                answer = StdSrvs_SetBool_Response.deserialize(
                    reply.ok.payload.to_bytes())
            except Exception as e:
                return False, f'the evacuation monitor reply: {type(e)}: {e}'
            return answer.success, f'evacuation monitor: {answer.message}'
        return False, 'the evacuation monitor did not answer'

    def _start_heartbeat(self):
        """
        Publish std_msgs/Empty on heartbeat_topic every heartbeat_period_sec.

        A node on the robot can watch this topic to find a lost link to the
        adapter, for example nav2_zone_manager with link_timeout. Without
        heartbeat_topic, there is no heartbeat.
        """
        topic = self.robot_config_yaml.get('heartbeat_topic', '')
        if not topic:
            return
        self.heartbeat_pub = self.zenoh_session.declare_publisher(
            namespacify(topic, self.name))
        payload = StdMsgs_Empty().serialize()
        self.heartbeat_timer = self.node.create_timer(
            self.robot_config_yaml.get('heartbeat_period_sec', 1.0),
            lambda: self.heartbeat_pub.put(payload))

    def _monotonic_sec(self) -> float:
        return time.monotonic()

    def _now_sec(self) -> float:
        seconds, nanoseconds = \
            self.node.get_clock().now().seconds_nanoseconds()
        return seconds + nanoseconds / 1e9

    def _trigger_replan(self):
        if self.replan_counts < self.MAX_REPLAN_COUNTS:
            self.replan_counts += 1
            self.node.get_logger().info(
                f'Replanning navigation for robot {self.name}, attempt {self.replan_counts}'
            )

            if self.update_handle is None:
                error_message = \
                    f'Failed to replan for robot {self.name}, robot ' \
                    'adapter has not yet been initialized with a fleet ' \
                    'update handle.'
                self.node.get_logger().error(error_message)
                return

            self.update_handle.more().replan()
        else:
            self.node.get_logger().error(
                f'Maximum replanning attempts reached for robot {self.name}'
            )
            self.replan_counts = 0
