#!/usr/bin/env python3
"""
ROS2 Node for Neural-Guided RRT* v2 path planning.

neural_rrt_star_node.py 대비 변경사항:
    - NeuralRRTStar import: neural_rrt_star_v2
    - 노드 이름: neural_rrt_star_node_v2
    - 토픽: /neural_rrt_star_v2/markers
    - 트리 색상: 초록색 (v1 보라색과 구분)
"""

import os
import csv
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from neural_rrt_planner.config import MAP_CONFIG
from neural_rrt_planner.planners.neural_rrt_star_v2 import NeuralRRTStar


class NeuralRRTStarNodeV2(Node):

    def __init__(self):
        super().__init__("neural_rrt_star_node_v2")

        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/neural_rrt_star_v2/markers",
            10,
        )

        self.frame_id = MAP_CONFIG["visualization"]["frame_id"]

        self.planner = NeuralRRTStar()

        self.iteration = 0
        self.max_iter = MAP_CONFIG["rrt_params"]["max_iter"]
        self.planning_finished = False

        # 질점 이동 관련
        self.agent_index: int = 0
        self.agent_moving: bool = False
        self.agent_tick: int = 0
        self.agent_tick_interval: int = 10  # 10틱(0.3s)마다 1 waypoint 이동

        self.get_logger().info("Neural-Guided RRT* v2 animated planning started.")

        self.timer = self.create_timer(0.03, self.timer_callback)

    # --------------------------------------------------
    # Timer Callback
    # --------------------------------------------------
    def timer_callback(self):
        if not self.planning_finished:
            self.planner.expand_once()
            self.iteration += 1

            if self.iteration >= self.max_iter:
                self.planning_finished = True
                if self.planner.best_goal_node is None:
                    self.get_logger().warn("Path not found within max_iter.")
                else:
                    self.get_logger().info(
                        f"Planning finished. "
                        f"Cost: {self.planner.best_goal_node.cost:.3f} | "
                        f"Waypoints: {len(self.planner.final_path)}"
                    )
                    for i, wp in enumerate(self.planner.final_path):
                        self.get_logger().info(
                            f"Waypoint {i:>2}: [{wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f}]"
                        )
                    self.save_waypoints_csv("neural_rrt_star_v2")
                    self.agent_moving = True
                    self.agent_index = 0

            if self.iteration % 50 == 0:
                best = (
                    f"{self.planner.best_goal_node.cost:.3f}"
                    if self.planner.best_goal_node else "None"
                )
                self.get_logger().info(
                    f"Iter: {self.iteration} | "
                    f"Nodes: {len(self.planner.nodes)} | "
                    f"Best cost: {best}"
                )

        # 질점 이동 (tick 기반 속도 조절)
        if self.agent_moving and self.planner.final_path:
            path = self.planner.final_path
            if self.agent_index < len(path):
                self.agent_tick += 1
                if self.agent_tick >= self.agent_tick_interval:
                    self.agent_tick = 0
                    self.agent_index += 1
            else:
                self.agent_moving = False
                self.get_logger().info("Agent reached goal.")

        self.publish_markers()

    # --------------------------------------------------
    # Save Waypoints CSV
    # --------------------------------------------------
    def save_waypoints_csv(self, planner_tag: str) -> None:
        output_dir = os.path.expanduser("~/term_project/waypoints")
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, f"waypoints_{planner_tag}.csv")

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "x", "y", "z"])
            for i, wp in enumerate(self.planner.final_path):
                writer.writerow([i, round(wp[0], 4), round(wp[1], 4), round(wp[2], 4)])

        self.get_logger().info(f"Waypoints saved: {filepath}")

    # --------------------------------------------------
    # Helper
    # --------------------------------------------------
    def make_point(self, position):
        point = Point()
        point.x = float(position[0])
        point.y = float(position[1])
        point.z = float(position[2])
        return point

    # --------------------------------------------------
    # Tree Marker (초록색 — v1 보라색과 구분)
    # --------------------------------------------------
    def create_tree_marker(self, marker_id):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "neural_rrt_tree_v2"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.01

        marker.color.r = 0.1
        marker.color.g = 0.9
        marker.color.b = 0.4
        marker.color.a = 0.6

        for node in self.planner.nodes:
            if node.parent is None:
                continue
            marker.points.append(self.make_point(node.position()))
            marker.points.append(self.make_point(node.parent.position()))

        return marker

    # --------------------------------------------------
    # Path Marker
    # --------------------------------------------------
    def create_path_marker(self, marker_id):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "final_path"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05

        marker.color.r = 1.0
        marker.color.g = 0.8
        marker.color.b = 0.0
        marker.color.a = 1.0

        for waypoint in self.planner.final_path:
            marker.points.append(self.make_point(waypoint))

        return marker

    # --------------------------------------------------
    # Sphere Marker (start / goal)
    # --------------------------------------------------
    def create_sphere_marker(self, marker_id, position, color, ns):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.12
        marker.scale.y = 0.12
        marker.scale.z = 0.12

        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])

        return marker

    # --------------------------------------------------
    # Obstacle Markers
    # --------------------------------------------------
    def create_obstacle_markers(self, start_id):
        markers = []
        marker_id = start_id

        for obs in MAP_CONFIG["obstacles"]:
            if obs["type"] != "cylinder":
                continue

            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "obstacles"
            marker.id = marker_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = float(obs["center"][0])
            marker.pose.position.y = float(obs["center"][1])
            marker.pose.position.z = float(obs["center"][2])
            marker.pose.orientation.w = 1.0

            marker.scale.x = float(obs["radius"] * 2.0)
            marker.scale.y = float(obs["radius"] * 2.0)
            marker.scale.z = float(obs["height"])

            marker.color.r = 0.8
            marker.color.g = 0.2
            marker.color.b = 0.2
            marker.color.a = 0.7

            markers.append(marker)
            marker_id += 1

        return markers

    # --------------------------------------------------
    # Map Boundary Marker
    # --------------------------------------------------
    def create_boundary_marker(self, marker_id):
        bounds = MAP_CONFIG["bounds"]

        x_min, x_max = bounds["x"]
        y_min, y_max = bounds["y"]
        z_min, z_max = bounds["z"]

        corners = [
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max],
        ]

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "map_boundary"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.015

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for i, j in edges:
            marker.points.append(self.make_point(corners[i]))
            marker.points.append(self.make_point(corners[j]))

        return marker

    # --------------------------------------------------
    # Agent Marker (질점)
    # --------------------------------------------------
    def create_agent_marker(self, marker_id):
        path = self.planner.final_path
        idx = min(self.agent_index, len(path) - 1)
        pos = path[idx]

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "agent"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(pos[0])
        marker.pose.position.y = float(pos[1])
        marker.pose.position.z = float(pos[2])
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15

        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 1.0

        return marker

    # --------------------------------------------------
    # Publish All Markers
    # --------------------------------------------------
    def publish_markers(self):
        marker_array = MarkerArray()
        marker_id = 0

        marker_array.markers.append(self.create_boundary_marker(marker_id))
        marker_id += 1

        marker_array.markers.append(self.create_tree_marker(marker_id))
        marker_id += 1

        if self.planner.final_path:
            marker_array.markers.append(self.create_path_marker(marker_id))
            marker_id += 1

        marker_array.markers.append(
            self.create_sphere_marker(
                marker_id, MAP_CONFIG["start"], [0.0, 1.0, 0.0, 1.0], "start"
            )
        )
        marker_id += 1

        marker_array.markers.append(
            self.create_sphere_marker(
                marker_id, MAP_CONFIG["goal"], [1.0, 0.0, 0.0, 1.0], "goal"
            )
        )
        marker_id += 1

        for marker in self.create_obstacle_markers(marker_id):
            marker_array.markers.append(marker)
            marker_id += 1

        # 질점 마커 — publish 전에 추가
        if self.planner.final_path and self.agent_index > 0:
            marker_array.markers.append(self.create_agent_marker(marker_id))
            marker_id += 1

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = NeuralRRTStarNodeV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# ROS2 Node for Neural-Guided RRT* v2 path planning.

# neural_rrt_star_node.py 대비 변경사항:
#     - NeuralRRTStar import: neural_rrt_star_v2
#     - 노드 이름: neural_rrt_star_node_v2
#     - 토픽: /neural_rrt_star_v2/markers
#     - 트리 색상: 초록색 (v1 보라색과 구분)
# """

# import os
# import csv
# import rclpy
# from rclpy.node import Node

# from geometry_msgs.msg import Point
# from visualization_msgs.msg import Marker, MarkerArray

# from neural_rrt_planner.config import MAP_CONFIG
# from neural_rrt_planner.planners.neural_rrt_star_v2 import NeuralRRTStar


# class NeuralRRTStarNodeV2(Node):

#     def __init__(self):
#         super().__init__("neural_rrt_star_node_v2")

#         self.marker_pub = self.create_publisher(
#             MarkerArray,
#             "/neural_rrt_star_v2/markers",
#             10,
#         )

#         self.frame_id = MAP_CONFIG["visualization"]["frame_id"]

#         self.planner = NeuralRRTStar()

#         self.iteration = 0
#         self.max_iter = MAP_CONFIG["rrt_params"]["max_iter"]
#         self.planning_finished = False

#         self.get_logger().info("Neural-Guided RRT* v2 animated planning started.")

#         self.timer = self.create_timer(
#             0.03,
#             self.timer_callback,
#         )
        
#         # 질점 이동 관련
#         self.agent_index: int = 0          # 현재 이동 중인 waypoint 인덱스
#         self.agent_moving: bool = False    # 이동 시작 여부
#         self.agent_tick: int = 0
#         self.agent_tick_interval: int = 10  # 10틱마다 1 waypoint 이동 (0.1s)
        

#     # --------------------------------------------------
#     # Timer Callback
#     # --------------------------------------------------
#     def timer_callback(self):
#         if not self.planning_finished:
#             self.planner.expand_once()
#             self.iteration += 1

#             if self.iteration >= self.max_iter:
#                 self.planning_finished = True
#                 if self.planner.best_goal_node is None:
#                     self.get_logger().warn("Path not found within max_iter.")
#                 else:
#                     self.get_logger().info(
#                         f"Planning finished. "
#                         f"Cost: {self.planner.best_goal_node.cost:.3f} | "
#                         f"Waypoints: {len(self.planner.final_path)}"
#                     )
#                     for i, wp in enumerate(self.planner.final_path):
#                         self.get_logger().info(
#                             f"Waypoint {i:>2}: [{wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f}]"
#                         )
#                     self.save_waypoints_csv("rrt_star")  # 각 파일에 맞는 태그
#                     self.agent_moving = True              # 이동 시작
#                     self.agent_index = 0

#             if self.iteration % 50 == 0:
#                 best = (
#                     f"{self.planner.best_goal_node.cost:.3f}"
#                     if self.planner.best_goal_node else "None"
#                 )
#                 self.get_logger().info(
#                     f"Iter: {self.iteration} | "
#                     f"Nodes: {len(self.planner.nodes)} | "
#                     f"Best cost: {best}"
#                 )

#         # 질점 이동
#         if self.agent_moving and self.planner.final_path:
#             path = self.planner.final_path
#             if self.agent_index < len(path):
#                 self.agent_index += 1
#             else:
#                 self.agent_moving = False
#                 self.get_logger().info("Agent reached goal.")

#         self.publish_markers()


#     def save_waypoints_csv(self, planner_tag: str) -> None:
#         output_dir = os.path.expanduser("~/term_project/waypoints")
#         os.makedirs(output_dir, exist_ok=True)

#         filepath = os.path.join(output_dir, f"waypoints_{planner_tag}.csv")

#         with open(filepath, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(["index", "x", "y", "z"])
#             for i, wp in enumerate(self.planner.final_path):
#                 writer.writerow([i, round(wp[0], 4), round(wp[1], 4), round(wp[2], 4)])

#         self.get_logger().info(f"Waypoints saved: {filepath}")

#     # --------------------------------------------------
#     # Helper
#     # --------------------------------------------------
#     def make_point(self, position):
#         point = Point()
#         point.x = float(position[0])
#         point.y = float(position[1])
#         point.z = float(position[2])
#         return point

#     # --------------------------------------------------
#     # Tree Marker (초록색 — v1 보라색과 구분)
#     # --------------------------------------------------
#     def create_tree_marker(self, marker_id):
#         marker = Marker()
#         marker.header.frame_id = self.frame_id
#         marker.header.stamp = self.get_clock().now().to_msg()
#         marker.ns = "neural_rrt_tree_v2"
#         marker.id = marker_id
#         marker.type = Marker.LINE_LIST
#         marker.action = Marker.ADD

#         marker.pose.orientation.w = 1.0
#         marker.scale.x = 0.01

#         marker.color.r = 0.1
#         marker.color.g = 0.9
#         marker.color.b = 0.4
#         marker.color.a = 0.6

#         for node in self.planner.nodes:
#             if node.parent is None:
#                 continue

#             marker.points.append(
#                 self.make_point(node.position())
#             )
#             marker.points.append(
#                 self.make_point(node.parent.position())
#             )

#         return marker

#     # --------------------------------------------------
#     # Path Marker
#     # --------------------------------------------------
#     def create_path_marker(self, marker_id):
#         marker = Marker()
#         marker.header.frame_id = self.frame_id
#         marker.header.stamp = self.get_clock().now().to_msg()
#         marker.ns = "final_path"
#         marker.id = marker_id
#         marker.type = Marker.LINE_STRIP
#         marker.action = Marker.ADD

#         marker.pose.orientation.w = 1.0
#         marker.scale.x = 0.05

#         marker.color.r = 1.0
#         marker.color.g = 0.8
#         marker.color.b = 0.0
#         marker.color.a = 1.0

#         for waypoint in self.planner.final_path:
#             marker.points.append(
#                 self.make_point(waypoint)
#             )

#         return marker

#     # --------------------------------------------------
#     # Sphere Marker (start / goal)
#     # --------------------------------------------------
#     def create_sphere_marker(self, marker_id, position, color, ns):
#         marker = Marker()
#         marker.header.frame_id = self.frame_id
#         marker.header.stamp = self.get_clock().now().to_msg()
#         marker.ns = ns
#         marker.id = marker_id
#         marker.type = Marker.SPHERE
#         marker.action = Marker.ADD

#         marker.pose.position.x = float(position[0])
#         marker.pose.position.y = float(position[1])
#         marker.pose.position.z = float(position[2])
#         marker.pose.orientation.w = 1.0

#         marker.scale.x = 0.12
#         marker.scale.y = 0.12
#         marker.scale.z = 0.12

#         marker.color.r = float(color[0])
#         marker.color.g = float(color[1])
#         marker.color.b = float(color[2])
#         marker.color.a = float(color[3])

#         return marker

#     # --------------------------------------------------
#     # Obstacle Markers
#     # --------------------------------------------------
#     def create_obstacle_markers(self, start_id):
#         markers = []
#         marker_id = start_id

#         for obs in MAP_CONFIG["obstacles"]:
#             if obs["type"] != "cylinder":
#                 continue

#             marker = Marker()
#             marker.header.frame_id = self.frame_id
#             marker.header.stamp = self.get_clock().now().to_msg()
#             marker.ns = "obstacles"
#             marker.id = marker_id
#             marker.type = Marker.CYLINDER
#             marker.action = Marker.ADD

#             marker.pose.position.x = float(obs["center"][0])
#             marker.pose.position.y = float(obs["center"][1])
#             marker.pose.position.z = float(obs["center"][2])
#             marker.pose.orientation.w = 1.0

#             marker.scale.x = float(obs["radius"] * 2.0)
#             marker.scale.y = float(obs["radius"] * 2.0)
#             marker.scale.z = float(obs["height"])

#             marker.color.r = 0.8
#             marker.color.g = 0.2
#             marker.color.b = 0.2
#             marker.color.a = 0.7

#             markers.append(marker)
#             marker_id += 1

#         return markers

#     # --------------------------------------------------
#     # Map Boundary Marker
#     # --------------------------------------------------
#     def create_boundary_marker(self, marker_id):
#         bounds = MAP_CONFIG["bounds"]

#         x_min, x_max = bounds["x"]
#         y_min, y_max = bounds["y"]
#         z_min, z_max = bounds["z"]

#         corners = [
#             [x_min, y_min, z_min],
#             [x_max, y_min, z_min],
#             [x_max, y_max, z_min],
#             [x_min, y_max, z_min],
#             [x_min, y_min, z_max],
#             [x_max, y_min, z_max],
#             [x_max, y_max, z_max],
#             [x_min, y_max, z_max],
#         ]

#         edges = [
#             (0, 1), (1, 2), (2, 3), (3, 0),
#             (4, 5), (5, 6), (6, 7), (7, 4),
#             (0, 4), (1, 5), (2, 6), (3, 7),
#         ]

#         marker = Marker()
#         marker.header.frame_id = self.frame_id
#         marker.header.stamp = self.get_clock().now().to_msg()
#         marker.ns = "map_boundary"
#         marker.id = marker_id
#         marker.type = Marker.LINE_LIST
#         marker.action = Marker.ADD

#         marker.pose.orientation.w = 1.0
#         marker.scale.x = 0.015

#         marker.color.r = 0.0
#         marker.color.g = 0.0
#         marker.color.b = 0.0
#         marker.color.a = 1.0

#         for i, j in edges:
#             marker.points.append(self.make_point(corners[i]))
#             marker.points.append(self.make_point(corners[j]))

#         return marker

#     # --------------------------------------------------
#     # Publish All Markers
#     # --------------------------------------------------
#     def publish_markers(self):
#         marker_array = MarkerArray()
#         marker_id = 0

#         marker_array.markers.append(
#             self.create_boundary_marker(marker_id)
#         )
#         marker_id += 1

#         marker_array.markers.append(
#             self.create_tree_marker(marker_id)
#         )
#         marker_id += 1

#         if self.planner.final_path:
#             marker_array.markers.append(
#                 self.create_path_marker(marker_id)
#             )
#             marker_id += 1

#         marker_array.markers.append(
#             self.create_sphere_marker(
#                 marker_id,
#                 MAP_CONFIG["start"],
#                 [0.0, 1.0, 0.0, 1.0],
#                 "start",
#             )
#         )
#         marker_id += 1

#         marker_array.markers.append(
#             self.create_sphere_marker(
#                 marker_id,
#                 MAP_CONFIG["goal"],
#                 [1.0, 0.0, 0.0, 1.0],
#                 "goal",
#             )
#         )
#         marker_id += 1

#         for marker in self.create_obstacle_markers(marker_id):
#             marker_array.markers.append(marker)
#             marker_id += 1

#         self.marker_pub.publish(marker_array)
        
#         # 질점 마커
#         if self.planner.final_path and self.agent_index > 0:
#             path = self.planner.final_path
#             idx = min(self.agent_index, len(path) - 1)
#             pos = path[idx]

#             agent_marker = Marker()
#             agent_marker.header.frame_id = "map"
#             agent_marker.header.stamp = self.get_clock().now().to_msg()
#             agent_marker.ns = "agent"
#             agent_marker.id = 9999
#             agent_marker.type = Marker.SPHERE
#             agent_marker.action = Marker.ADD
#             agent_marker.pose.position.x = pos[0]
#             agent_marker.pose.position.y = pos[1]
#             agent_marker.pose.position.z = pos[2]
#             agent_marker.pose.orientation.w = 1.0
#             agent_marker.scale.x = 0.12
#             agent_marker.scale.y = 0.12
#             agent_marker.scale.z = 0.12
#             agent_marker.color.r = 1.0
#             agent_marker.color.g = 0.5
#             agent_marker.color.b = 0.0
#             agent_marker.color.a = 1.0

#             marker_array.markers.append(agent_marker)


# def main(args=None):
#     rclpy.init(args=args)

#     node = NeuralRRTStarNodeV2()

#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()

# if __name__ == "__main__":
#     main()