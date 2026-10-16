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
import threading

from free_fleet.ros2_types import Header
from free_fleet.utils import namespacify
import pycdr2
from pycdr2 import IdlStruct
import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rmf_adapter.robot_update_handle import Tier
from rmf_zone_manager_msgs.msg import FleetZoneState
from rmf_zone_manager_msgs.msg import ZoneStates as RmfZoneStates
import zenoh


# CDR types matching nav2_zone_manager/srv/SetZoneState and
# nav2_zone_manager/srv/GetZoneStates of nav2_zone_evacuation
@dataclass
class SetZoneState_Request(IdlStruct):
    name: str
    enabled: bool


@dataclass
class SetZoneState_Response(IdlStruct):
    success: bool
    message: str


@dataclass
class GetZoneStates_Request(IdlStruct):
    # rosidl adds this member to an empty request
    structure_needs_at_least_one_member: pycdr2.types.uint8 = 0


# https://github.com/ros2/common_interfaces/blob/rolling/geometry_msgs/msg/Point32.msg
@dataclass
class GeometryMsgs_Point32(IdlStruct):
    x: pycdr2.types.float32
    y: pycdr2.types.float32
    z: pycdr2.types.float32


# https://github.com/ros2/common_interfaces/blob/rolling/geometry_msgs/msg/Polygon.msg
@dataclass
class GeometryMsgs_Polygon(IdlStruct):
    points: pycdr2.types.sequence[GeometryMsgs_Point32]


@dataclass
class ZoneState(IdlStruct):
    name: str
    enabled: bool
    zone_type: str
    priority: pycdr2.types.uint8
    description: str
    polygon: GeometryMsgs_Polygon
    evacuate_on_link_loss: bool


@dataclass
class ZoneStates(IdlStruct):
    header: Header
    zones: pycdr2.types.sequence[ZoneState]


@dataclass
class GetZoneStates_Response(IdlStruct):
    zone_states: ZoneStates


@dataclass
class ZoneSyncActions:
    enable: set[str]
    adopt: set[str]
    disable: set[str]
    forget: set[str]
    foreign: set[str]


def zone_sync_actions(
    required: set[str], on: set[str], recorded: set[str]
) -> ZoneSyncActions:
    """
    Compare the zones that RMF requires with the zones that the robot has on.

    required: the zones that RMF closed with evacuation LEAVE_NOW.
    on: the zones that the robot reports as enabled.
    recorded: the zones that the sync turned on or adopted.
    """
    not_required = recorded - required
    return ZoneSyncActions(
        enable=required - on,
        adopt=required & on,
        disable=not_required & on,
        forget=not_required - on,
        foreign=on - required - recorded,
    )


@dataclass
class _RobotZones:
    name: str
    update_handle: object
    recorded: set[str] = field(default_factory=set)
    reachable: bool = True
    foreign_issue: object = None
    foreign_issue_zones: set[str] = field(default_factory=set)
    failed_issue: object = None
    failed_issue_message: str = ''


class Nav2ZoneSync:
    """
    Turn the robot zones on and off as the RMF zone closures require.

    A zone that RMF closed with evacuation LEAVE_NOW is turned on in the
    nav2_zone_manager of each robot of the fleet. The sync turns off only the
    zones that it turned on or adopted. The get_zone_states call to each robot
    runs every period_sec and after each change of zone_states.
    """

    def __init__(
        self,
        zenoh_session: zenoh.Session,
        node,
        fleet_name: str,
        service_call_timeout_sec: float = 1.0,
        period_sec: float = 5.0,
    ):
        self.zenoh_session = zenoh_session
        self.node = node
        self.fleet_name = fleet_name
        self.service_call_timeout_sec = service_call_timeout_sec
        self.period_sec = period_sec
        self.mutex = threading.Lock()
        # None until the first zone_states message
        self.required: set[str] | None = None
        self.robots: list[_RobotZones] = []
        self.wake = threading.Event()

    def add_robot(self, robot_name: str, update_handle):
        with self.mutex:
            self.robots.append(_RobotZones(robot_name, update_handle))

    def start(self):
        # As the zone manager publishes zone_states
        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.subscription = self.node.create_subscription(
            RmfZoneStates, 'zone_states', self.zone_states_callback, qos)
        # The Zenoh gets block. Thus they do not run on the executor.
        threading.Thread(target=self._run, daemon=True).start()

    def zone_states_callback(self, msg: RmfZoneStates):
        required = {
            zone.name
            for zone in msg.zones
            for fleet in zone.fleets
            if fleet.fleet_name == self.fleet_name and fleet.closed and
            fleet.evacuation == FleetZoneState.EVACUATION_LEAVE_NOW
        }
        with self.mutex:
            self.required = required
        self.wake.set()

    def _run(self):
        while rclpy.ok():
            self.wake.wait(self.period_sec)
            self.wake.clear()
            try:
                self.sync()
            except Exception as e:
                self.node.get_logger().error(
                    f'Zone sync failed: {type(e)}: {e}')

    def sync(self):
        with self.mutex:
            required = self.required
            robots = list(self.robots)
        for robot in robots:
            self._sync_robot(robot, required)

    def _sync_robot(self, robot: _RobotZones, required: set[str] | None):
        on = self._get_enabled_zones(robot)
        if on is None:
            self._update_failed_issue(
                robot, sorted(required or []),
                'Evacuation not applied: the robot does not answer. The zone '
                'sync cannot turn on the zones')
            return
        if required is None:
            return
        actions = zone_sync_actions(required, on, robot.recorded)
        failed = []
        for zone in sorted(actions.enable):
            if self._set_zone(robot, zone, True):
                robot.recorded.add(zone)
            else:
                failed.append(zone)
        robot.recorded |= actions.adopt
        for zone in sorted(actions.disable):
            if self._set_zone(robot, zone, False):
                robot.recorded.discard(zone)
            else:
                failed.append(zone)
        robot.recorded -= actions.forget
        self._update_foreign_issue(robot, actions.foreign)
        self._update_failed_issue(
            robot, failed,
            'Evacuation not applied: the zone sync could not set the zones')

    def _get_enabled_zones(self, robot: _RobotZones) -> set[str] | None:
        response = self._call(
            robot, 'get_zone_states', GetZoneStates_Request(),
            GetZoneStates_Response)
        if response is None:
            if robot.reachable:
                self.node.get_logger().warn(
                    f'Zone sync: no zone states from robot [{robot.name}]')
            robot.reachable = False
            return None
        if not robot.reachable:
            self.node.get_logger().info(
                f'Zone sync: zone states from robot [{robot.name}] again')
        robot.reachable = True
        return {
            zone.name for zone in response.zone_states.zones if zone.enabled}

    def _set_zone(self, robot: _RobotZones, zone: str, enabled: bool) -> bool:
        response = self._call(
            robot, 'set_zone_state', SetZoneState_Request(zone, enabled),
            SetZoneState_Response)
        success = response is not None and response.success
        message = 'no reply' if response is None else response.message
        log = self.node.get_logger().info if success else \
            self.node.get_logger().error
        log(f'Zone sync: set zone [{zone}] of robot [{robot.name}] to '
            f'{"on" if enabled else "off"}: success {success}, {message}')
        return success

    def _call(self, robot: _RobotZones, service: str, request, response_type):
        """Call a service of the robot. Return None if there is no reply."""
        try:
            for reply in self.zenoh_session.get(
                namespacify(service, robot.name),
                payload=request.serialize(),
                timeout=self.service_call_timeout_sec,
            ):
                if reply.ok is None:
                    return None
                return response_type.deserialize(reply.ok.payload.to_bytes())
        except Exception as e:
            self.node.get_logger().warn(
                f'Zone sync: {service} of robot [{robot.name}] failed: '
                f'{type(e)}: {e}')
        return None

    def _update_foreign_issue(self, robot: _RobotZones, foreign: set[str]):
        if robot.foreign_issue is not None and \
                foreign != robot.foreign_issue_zones:
            robot.foreign_issue.resolve({})
            robot.foreign_issue = None
        if foreign and robot.foreign_issue is None:
            message = (
                f'Robot [{robot.name}] has zones on that RMF does not '
                f'require: {sorted(foreign)}. The adapter does not turn them '
                'off. Turn them off on the robot, or close them in RMF with '
                'evacuation leave_now and then open them.'
            )
            self.node.get_logger().warn(message)
            robot.foreign_issue_zones = set(foreign)
            robot.foreign_issue = robot.update_handle.more().create_issue(
                Tier.Warning, 'foreign_robot_zones',
                {'message': message, 'zones': sorted(foreign)})

    def _update_failed_issue(
        self, robot: _RobotZones, failed: list[str], reason: str
    ):
        message = (
            f'{reason} {failed} of robot [{robot.name}]. It tries again.'
            if failed else '')
        if robot.failed_issue is not None and \
                message != robot.failed_issue_message:
            robot.failed_issue.resolve({})
            robot.failed_issue = None
        if failed and robot.failed_issue is None:
            self.node.get_logger().warn(message)
            robot.failed_issue_message = message
            robot.failed_issue = robot.update_handle.more().create_issue(
                Tier.Error, 'evacuation_not_applied',
                {'message': message, 'zones': failed})
