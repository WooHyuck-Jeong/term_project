#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from neural_rrt_planner.config import MAP_CONFIG
from neural_rrt_planner.planners.informed_rrt_star import InformedRRTStar


class InformedRRTStarNode(Node):
    def __init__(self):
        super().__init__("informed_rrt_star_node")

        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/informed_rrt_star/markers",
            10
        )

        self.frame_id = MAP_CONFIG["visualization"]["frame_id"]

        self.planner = InformedRRTStar()

        self.iteration = 0
        self.max_iter = MAP_CONFIG["rrt_params"]["max_iter"]

        self.planning_finished = False

        self.get_logger().info("Informed RRT* animated planning started.")

        self.timer = self.create_timer(
            0.03,
            self.timer_callback
        )

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

        self.publish_markers()

    def make_point(self, position):
        point = Point()
        point.x = float(position[0])
        point.y = float(position[1])
        point.z = float(position[2])
        return point

    def create_tree_marker(self, marker_id):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "informed_rrt_tree"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.01

        marker.color.r = 0.1
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 0.6

        for node in self.planner.nodes:
            if node.parent is None:
                continue

            marker.points.append(
                self.make_point(node.position())
            )
            marker.points.append(
                self.make_point(node.parent.position())
            )

        return marker

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
            marker.points.append(
                self.make_point(waypoint)
            )

        return marker

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

    def create_boundary_marker(self, marker_id):
        bounds = MAP_CONFIG["bounds"]

        x_min, x_max = bounds["x"]
        y_min, y_max = bounds["y"]
        z_min, z_max = bounds["z"]

        corners = [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
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
            marker.points.append(
                self.make_point(corners[i])
            )
            marker.points.append(
                self.make_point(corners[j])
            )

        return marker

    def quaternion_from_x_axis_to_vector(self, vector):
        v = np.array(vector, dtype=float)
        norm = np.linalg.norm(v)

        if norm < 1e-9:
            return [0.0, 0.0, 0.0, 1.0]

        v = v / norm
        x_axis = np.array([1.0, 0.0, 0.0])

        cross = np.cross(x_axis, v)
        dot = np.dot(x_axis, v)

        if dot < -0.999999:
            return [0.0, 0.0, 1.0, 0.0]

        q = np.array([
            cross[0],
            cross[1],
            cross[2],
            1.0 + dot
        ])

        q = q / np.linalg.norm(q)
        return q.tolist()

    def create_ellipsoid_marker(self, marker_id):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "informed_ellipsoid"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        ellipsoid = self.planner.get_ellipsoid_info()

        if ellipsoid is None:
            marker.action = Marker.DELETE
            return marker

        center = ellipsoid["center"]
        axes = ellipsoid["axes"]
        direction = ellipsoid["direction"]

        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2])

        q = self.quaternion_from_x_axis_to_vector(direction)

        marker.pose.orientation.x = float(q[0])
        marker.pose.orientation.y = float(q[1])
        marker.pose.orientation.z = float(q[2])
        marker.pose.orientation.w = float(q[3])

        marker.scale.x = float(2.0 * axes[0])
        marker.scale.y = float(2.0 * axes[1])
        marker.scale.z = float(2.0 * axes[2])

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.3
        marker.color.a = 0.18

        return marker

    # def create_iteration_text_marker(self, marker_id):
    #     marker = Marker()
    #     marker.header.frame_id = self.frame_id
    #     marker.header.stamp = self.get_clock().now().to_msg()
    #     marker.ns = "iteration_text"
    #     marker.id = marker_id
    #     marker.type = Marker.TEXT_VIEW_FACING
    #     marker.action = Marker.ADD

    #     marker.pose.position.x = 0.2
    #     marker.pose.position.y = 0.1
    #     marker.pose.position.z = 0.2
    #     marker.pose.orientation.w = 1.0

    #     marker.scale.z = 0.18

    #     marker.color.r = 1.0
    #     marker.color.g = 1.0
    #     marker.color.b = 1.0
    #     marker.color.a = 1.0

    #     if self.planner.best_goal_node is None:
    #         best_cost = "None"
    #     else:
    #         best_cost = f"{self.planner.best_goal_node.cost:.3f}"

    #     marker.text = (
    #         f"Informed RRT* | Iter: {self.iteration} | "
    #         f"Nodes: {len(self.planner.nodes)} | "
    #         f"Best cost: {best_cost}"
    #     )

    #     return marker

    def publish_markers(self):
        marker_array = MarkerArray()
        marker_id = 0

        marker_array.markers.append(
            self.create_boundary_marker(marker_id)
        )
        marker_id += 1

        marker_array.markers.append(
            self.create_tree_marker(marker_id)
        )
        marker_id += 1

        if self.planner.final_path:
            marker_array.markers.append(
                self.create_path_marker(marker_id)
            )
            marker_id += 1

        marker_array.markers.append(
            self.create_sphere_marker(
                marker_id,
                MAP_CONFIG["start"],
                [0.0, 1.0, 0.0, 1.0],
                "start"
            )
        )
        marker_id += 1

        marker_array.markers.append(
            self.create_sphere_marker(
                marker_id,
                MAP_CONFIG["goal"],
                [1.0, 0.0, 0.0, 1.0],
                "goal"
            )
        )
        marker_id += 1

        for marker in self.create_obstacle_markers(marker_id):
            marker_array.markers.append(marker)
            marker_id += 1

        marker_array.markers.append(
            self.create_ellipsoid_marker(marker_id)
        )
        marker_id += 1

        # marker_array.markers.append(
        #     self.create_iteration_text_marker(marker_id)
        # )

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)

    node = InformedRRTStarNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()