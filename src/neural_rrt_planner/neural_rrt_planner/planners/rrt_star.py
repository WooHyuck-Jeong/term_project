#! /usr/bin/env python3
import math
import random

from neural_rrt_planner.config import MAP_CONFIG
from neural_rrt_planner.utils.collision_checker import CollisionChecker
from neural_rrt_planner.planners.rrt_node import RRTNode


class RRTStar:

    def __init__(self):
        self.map_config = MAP_CONFIG
        self.rrt_params = MAP_CONFIG["rrt_params"]

        self.collision_checker = CollisionChecker()

        start = self.map_config["start"]
        goal = self.map_config["goal"]

        self.start_node = RRTNode(start[0], start[1], start[2])
        self.goal_node = RRTNode(goal[0], goal[1], goal[2])

        self.nodes = [self.start_node]

        self.best_goal_node = None
        self.final_path = []

        self.step_size = self.rrt_params["step_size"]
        self.goal_sample_rate = self.rrt_params["goal_sample_rate"]
        self.max_iter = self.rrt_params["max_iter"]
        self.search_radius = self.rrt_params["search_radius"]
        self.goal_threshold = self.rrt_params["goal_threshold"]

    # --------------------------------------------------
    # Distance
    # --------------------------------------------------
    def distance(self, node1, node2):
        return math.sqrt(
            (node1.x - node2.x) ** 2
            +
            (node1.y - node2.y) ** 2
            +
            (node1.z - node2.z) ** 2
        )

    # --------------------------------------------------
    # Random Sampling
    # --------------------------------------------------
    def sample_random_node(self):
        if random.random() < self.goal_sample_rate:
            return RRTNode(
                self.goal_node.x,
                self.goal_node.y,
                self.goal_node.z
            )

        bounds = self.map_config["bounds"]

        while True:
            x = random.uniform(
                bounds["x"][0],
                bounds["x"][1]
            )

            y = random.uniform(
                bounds["y"][0],
                bounds["y"][1]
            )

            z = random.uniform(
                bounds["z"][0],
                bounds["z"][1]
            )

            node = RRTNode(x, y, z)

            if self.collision_checker.is_valid_point(
                node.position()
            ):
                return node

    # --------------------------------------------------
    # Nearest Node
    # --------------------------------------------------
    def nearest_node(self, random_node):
        return min(
            self.nodes,
            key=lambda node: self.distance(
                node,
                random_node
            )
        )

    # --------------------------------------------------
    # Steer
    # --------------------------------------------------
    def steer(self, from_node, to_node):
        dist = self.distance(
            from_node,
            to_node
        )

        if dist <= self.step_size:
            new_node = RRTNode(
                to_node.x,
                to_node.y,
                to_node.z
            )
        else:
            ratio = self.step_size / dist

            new_node = RRTNode(
                from_node.x
                +
                (to_node.x - from_node.x) * ratio,

                from_node.y
                +
                (to_node.y - from_node.y) * ratio,

                from_node.z
                +
                (to_node.z - from_node.z) * ratio,
            )

        new_node.parent = from_node
        new_node.cost = (
            from_node.cost
            +
            self.distance(from_node, new_node)
        )

        return new_node

    # --------------------------------------------------
    # Find Near Nodes
    # --------------------------------------------------
    def find_near_nodes(self, new_node):
        near_nodes = []

        for node in self.nodes:
            if self.distance(node, new_node) <= self.search_radius:
                near_nodes.append(node)

        return near_nodes

    # --------------------------------------------------
    # Choose Best Parent
    # --------------------------------------------------
    def choose_parent(self, new_node, near_nodes):
        best_parent = new_node.parent
        best_cost = new_node.cost

        for near_node in near_nodes:
            if not self.collision_checker.is_collision_free(
                near_node.position(),
                new_node.position()
            ):
                continue

            candidate_cost = (
                near_node.cost
                +
                self.distance(near_node, new_node)
            )

            if candidate_cost < best_cost:
                best_parent = near_node
                best_cost = candidate_cost

        new_node.parent = best_parent
        new_node.cost = best_cost

        return new_node

    # --------------------------------------------------
    # Rewire
    # --------------------------------------------------
    def rewire(self, new_node, near_nodes):
        for near_node in near_nodes:
            if near_node == new_node.parent:
                continue

            if not self.collision_checker.is_collision_free(
                new_node.position(),
                near_node.position()
            ):
                continue

            new_cost = (
                new_node.cost
                +
                self.distance(new_node, near_node)
            )

            if new_cost < near_node.cost:
                near_node.parent = new_node
                near_node.cost = new_cost

    # --------------------------------------------------
    # Check Goal Connection
    # --------------------------------------------------
    def check_goal_connection(self, new_node):
        distance_to_goal = self.distance(
            new_node,
            self.goal_node
        )

        if distance_to_goal > self.goal_threshold:
            return

        if not self.collision_checker.is_collision_free(
            new_node.position(),
            self.goal_node.position()
        ):
            return

        candidate_goal_node = RRTNode(
            self.goal_node.x,
            self.goal_node.y,
            self.goal_node.z
        )

        candidate_goal_node.parent = new_node
        candidate_goal_node.cost = (
            new_node.cost
            +
            self.distance(new_node, candidate_goal_node)
        )

        if (
            self.best_goal_node is None
            or
            candidate_goal_node.cost < self.best_goal_node.cost
        ):
            self.best_goal_node = candidate_goal_node
            self.final_path = self.extract_path(candidate_goal_node)

            print(
                "New best path found | "
                f"Cost: {candidate_goal_node.cost:.3f} | "
                f"Waypoints: {len(self.final_path)}"
            )

    # --------------------------------------------------
    # Extract Path
    # --------------------------------------------------
    def extract_path(self, goal_node):
        path = []
        node = goal_node

        while node is not None:
            path.append(node.position())
            node = node.parent

        path.reverse()
        return path

    # --------------------------------------------------
    # One RRT* Expansion Step
    # --------------------------------------------------
    def expand_once(self):
        random_node = self.sample_random_node()

        nearest = self.nearest_node(random_node)

        new_node = self.steer(
            nearest,
            random_node
        )

        if not self.collision_checker.is_collision_free(
            nearest.position(),
            new_node.position()
        ):
            return False

        near_nodes = self.find_near_nodes(new_node)

        new_node = self.choose_parent(
            new_node,
            near_nodes
        )

        self.nodes.append(new_node)

        self.rewire(
            new_node,
            near_nodes
        )

        self.check_goal_connection(new_node)

        return True

    # --------------------------------------------------
    # Planning
    # --------------------------------------------------
    def plan(self):
        success_count = 0

        for i in range(self.max_iter):
            success = self.expand_once()

            if success:
                success_count += 1

            if i % 100 == 0:
                if self.best_goal_node is None:
                    print(
                        f"Iteration: {i}, "
                        f"Nodes: {len(self.nodes)}, "
                        f"Success: {success_count}, "
                        f"Best cost: None"
                    )
                else:
                    print(
                        f"Iteration: {i}, "
                        f"Nodes: {len(self.nodes)}, "
                        f"Success: {success_count}, "
                        f"Best cost: {self.best_goal_node.cost:.3f}"
                    )

        print("Planning finished")
        print(f"Total nodes: {len(self.nodes)}")

        if self.best_goal_node is not None:
            print(f"Final path cost: {self.best_goal_node.cost:.3f}")
            print(f"Waypoint count: {len(self.final_path)}")
        else:
            print("Path not found")

        return self.final_path



''' rrt star base line '''

# import math
# import random

# from neural_rrt_planner.config import MAP_CONFIG
# from neural_rrt_planner.utils.collision_checker import CollisionChecker
# from neural_rrt_planner.planners.rrt_node import RRTNode

# class RRTStar:
#     def __init__(self):
#         self.map_config = MAP_CONFIG
#         self.rrt_params = MAP_CONFIG['rrt_params']

#         self.collision_checker = CollisionChecker()
        
#         start = self.map_config['start']
#         goal = self.map_config['goal']

#         self.start_node = RRTNode(start[0], start[1], start[2])
#         self.goal_node = RRTNode(goal[0], goal[1], goal[2])

#         self.nodes = [self.start_node]

#         self.step_size = self.rrt_params['step_size']
#         self.goal_sample_rate = self.rrt_params['goal_sample_rate']
#         self.max_iter = self.rrt_params['max_iter']

#     # === Distance ===
#     def distance(self, node1, node2):
#         return math.sqrt(
#             (node1.x - node2.x) ** 2
#             +
#             (node1.y - node2.y) ** 2
#             +
#             (node1.z - node2.z) ** 2
#         )
        
#     # === Random Sampling ===
#     def sample_random_node(self):
#         if random.random() < self.goal_sample_rate:
#             return RRTNode(
#                 self.goal_node.x,
#                 self.goal_node.y,
#                 self.goal_node.z
#             )
        
#         bounds = self.map_config['bounds']

#         x = random.uniform(
#             bounds['x'][0],
#             bounds['x'][1]
#         )
#         y = random.uniform(
#             bounds['y'][0],
#             bounds['y'][1]
#         )
#         z = random.uniform(
#             bounds['z'][0],
#             bounds['z'][1]
#         )
        
#         return RRTNode(x, y, z)

#     # === Nearest Node ===
#     def nearest_node(self, random_node):
#         return min(
#             self.nodes,
#             key = lambda node: self.distance(
#                 node,
#                 random_node
#             )
#         )
        
#     # === Steer ===
#     def steer(self, from_node, to_node):
#         dist = self.distance(
#             from_node,
#             to_node
#         )
        
#         if dist <= self.step_size:
#             new_node = RRTNode(
#                 to_node.x,
#                 to_node.y,
#                 to_node.z
#             )
#         else:
#             ratio = self.step_size / dist
            
#             new_node = RRTNode(
#                 from_node.x
#                 +
#                 (to_node.x - from_node.x) * ratio,
                
#                 from_node.y
#                 +
#                 (to_node.y - from_node.y) * ratio,
                
#                 from_node.z
#                 +
#                 (to_node.z - from_node.z) * ratio,
#             )
        
#         new_node.parent = from_node
#         new_node.cost = (
#             from_node.cost + self.distance(from_node, new_node)
#         )
        
#         return new_node
    
#     # === One RRT Expansion Step ===
#     def expand_once(self):
#         random_node = self.sample_random_node()
        
#         nearest = self.nearest_node(random_node)

#         new_node = self.steer(
#             nearest,
#             random_node
#         )
        
#         if self.collision_checker.is_collision_free(
#             nearest.position(),
#             new_node.position()
#         ):
#             self.nodes.append(new_node)
#             return True
        
#         return False
    
#     # === Planning ===
#     def plan(self):
#         success_count = 0
        
#         for i in range(self.max_iter):
#             success = self.expand_once()
            
#             if success:
#                 success_count += 1
                
#                 if i % 100 == 0:
#                     print(
#                         f'Iteration: {i}, '
#                         f'Nodes: {len(self.nodes)}, '
#                         f'Success: {success_count}'
#                     )
                    
            
#         print('Planning finished')
#         print(f'Total nodes: {len(self.nodes)}')

#         return self.nodes
    
    