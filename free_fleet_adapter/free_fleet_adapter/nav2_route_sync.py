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

from dataclasses import dataclass

from free_fleet.utils import namespacify
import pycdr2
from pycdr2 import IdlStruct
import zenoh


# CDR types matching nav2_msgs/msg/EdgeCost and nav2_msgs/srv/DynamicEdges
# https://github.com/ros-navigation/navigation2/blob/main/nav2_msgs/msg/EdgeCost.msg
@dataclass
class EdgeCost(IdlStruct):
    edgeid: pycdr2.types.uint16
    cost: pycdr2.types.float32


# https://github.com/ros-navigation/navigation2/blob/main/nav2_msgs/srv/DynamicEdges.srv
@dataclass
class DynamicEdges_Request(IdlStruct):
    closed_edges: pycdr2.types.sequence[pycdr2.types.uint16]
    opened_edges: pycdr2.types.sequence[pycdr2.types.uint16]
    adjust_edges: pycdr2.types.sequence[EdgeCost]


@dataclass
class DynamicEdges_Response(IdlStruct):
    success: bool


class Nav2RouteSync:
    """Synchronizes RMF lane closures to Nav2 route servers.

    Sends DynamicEdges service requests over Zenoh to each robot's
    DynamicEdgesScorer plugin.
    """

    def __init__(
        self,
        zenoh_session: zenoh.Session,
        node,
        route_server_name: str = 'route_server',
        scorer_name: str = 'DynamicEdgesScorer',
        service_call_timeout_sec: float = 1.0,
    ):
        self.zenoh_session = zenoh_session
        self.node = node
        self.service_call_timeout_sec = service_call_timeout_sec
        self.service_key = f'{route_server_name}/{scorer_name}/adjust_edges'
        self.robot_names: list[str] = []
        self.closed_lanes: set[int] = set()

    def add_robot(self, robot_name: str):
        self.robot_names.append(robot_name)
        if self.closed_lanes:
            self._send_to_robot(
                robot_name,
                closed_edges=list(self.closed_lanes),
                opened_edges=[],
                adjust_edges=[],
            )

    def sync_lane_closures(
        self,
        open_lanes: list[int],
        close_lanes: list[int],
    ):
        for lane_idx in close_lanes:
            self.closed_lanes.add(lane_idx)
        for lane_idx in open_lanes:
            self.closed_lanes.discard(lane_idx)

        self._send_to_all(
            closed_edges=close_lanes,
            opened_edges=open_lanes,
            adjust_edges=[],
        )

    def _send_to_all(self, closed_edges, opened_edges, adjust_edges):
        for robot_name in self.robot_names:
            self._send_to_robot(
                robot_name, closed_edges, opened_edges, adjust_edges
            )

    def _send_to_robot(
        self,
        robot_name: str,
        closed_edges: list[int],
        opened_edges: list[int],
        adjust_edges: list[EdgeCost],
    ):
        req = DynamicEdges_Request(
            closed_edges=closed_edges,
            opened_edges=opened_edges,
            adjust_edges=adjust_edges,
        )
        key = namespacify(self.service_key, robot_name)
        try:
            replies = self.zenoh_session.get(
                key,
                payload=req.serialize(),
                timeout=self.service_call_timeout_sec,
            )
            for reply in replies:
                try:
                    resp = DynamicEdges_Response.deserialize(
                        reply.ok.payload.to_bytes()
                    )
                    if resp.success:
                        self.node.get_logger().info(
                            f'Route sync to [{robot_name}] succeeded '
                            f'(closed={closed_edges}, opened={opened_edges}, '
                            f'adjust={len(adjust_edges)})'
                        )
                    else:
                        self.node.get_logger().warn(
                            f'Route sync to [{robot_name}] returned '
                            f'success=false'
                        )
                except Exception as e:
                    self.node.get_logger().warn(
                        f'Failed to parse route sync reply from '
                        f'[{robot_name}]: {e}'
                    )
        except Exception as e:
            self.node.get_logger().warn(
                f'Route sync Zenoh call to [{robot_name}] failed: {e}'
            )
