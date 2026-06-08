#! /usr/bin/env python3

import csv
import math
import os
import random
from typing import TypeAlias

from neural_rrt_planner.config import MAP_CONFIG
from neural_rrt_planner.utils.collision_checker import CollisionChecker

Point3D: TypeAlias = list[float]
Direction3D: TypeAlias = list[float]


class SamplingDatasetGenerator:
    """
    Generate sampling dataset for learning
    
    Input:
        current_x, current_y, current_z
        goal_x, goal_y, goal_z,
        nearest_obstacle_distance
        
    Target:
        target_dx, target_dy, target_dz
        
    Target Direction:
        current ~ goal: unit direction vector
    """
    
    def __init__(self) -> None:
        self.map_config: dict = MAP_CONFIG
        self.bounds: dict = MAP_CONFIG['bounds']
        self.goal: Point3D = MAP_CONFIG['goal']
        self.obstacles: list[dict] = MAP_CONFIG['obstacles']
        self.depth_limit: dict = MAP_CONFIG['depth_limit']

        self.collision_checker = CollisionChecker()
        self.dataset_dir: str = os.path.expanduser(
            "~/term_project/datasets"
        )
        os.makedirs(
            self.dataset_dir,
            exist_ok= True
        )
        
        self.dataset_path = os.path.join(
            self.dataset_dir,
            "sampling_dataset.csv"
        )
        
        
    def sample_valid_point(self) -> Point3D:
        """
        Generate a random current position that does not collide with anything in the map
        
        Returns:
            Point3D:
                [x, y, z] valid 3D position
        """
        
        while True:
            x: float = random.uniform(
                self.bounds['x'][0],
                self.bounds['x'][1]
            )
            y: float = random.uniform(
                self.bounds['y'][0],
                self.bounds['y'][1]
            )
            z: float = random.uniform(
                self.bounds['z'][0],
                self.bounds['z'][1]
            )
            
            point: Point3D = [x, y, z]

            if self.collision_checker.is_valid_point(point):
                return point
            
            
    def compute_goal_direction(
        self,
        current: Point3D
    ) -> Direction3D:
        """
        Caculate the unit direction vector from the current position toward the goal.
        
        Args:
            current (Point3D):
                current position [x, y, z]
                
        Returns
            Direction3D:
                unit vector [dx, dy, dz] for goal direction
        """
        
        dx: float = self.goal[0] - current[0]
        dy: float = self.goal[1] - current[1]
        dz: float = self.goal[2] - current[2]

        norm: float = math.sqrt(
            dx ** 2 + dy** 2 + dz ** 2
        )
        
        if norm < 1e-9:
            return [0.0, 0.0, 0.0]

        return [
            dx / norm,
            dy / norm,
            dz / norm
        ]
        
        
    def compute_nearest_obstacle_distance(
        self,
        point: Point3D,
    ) -> float:
        """
        Calculate the distance from the current point to the nearest cylinder obstacle.
        
        The distance is calculated including the safety margin.
        
        Args:
            point (Point3D):
               Position for distance calculation
               
        Returns:
            float:
                Distance to the nearest obstacle[m]
                If there are no obstacles, returns the maximum value of 10.0
        """
        
        min_distance: float = float('inf')

        px: float = point[0]
        py: float = point[1]
        pz: float = point[2]

        safety_margin: float = self.map_config['safety_margin']

        for obstacle in self.obstacles:
            if obstacle['type'] != 'cylinder':
                continue
            
            cx: float = obstacle['center'][0]
            cy: float = obstacle['center'][1]
            cz: float = obstacle['center'][2]

            radius: float = obstacle['radius'] + safety_margin
            height: float = obstacle['height'] + 2.0 * safety_margin
            
            dx: float = px - cx
            dy: float = py - cy

            horizontal_distance: float = math.sqrt(dx ** 2 + dy ** 2) - radius
            
            z_min: float = cz - height / 2.0
            z_max: float = cz + height / 2.0
            
            if pz < z_min:
                vertical_distance: float = z_min - pz
            elif pz > z_max:
                vertical_distance = pz - z_max
            else:
                vertical_distance = 0.0
                
            horizontal_distance = max(
                horizontal_distance,
                0.0
            )
            
            distance: float = math.sqrt(
                horizontal_distance ** 2 + vertical_distance ** 2
            )
            
            min_distance = min(
                min_distance,
                distance
            )        
            
        if min_distance == float('inf'):
            return 10.0
        
        return min_distance
    
    
    def create_dataset_row(self, current: Point3D) -> list[float]:
        """
        Create a CSV row from a single current point
        
        Args:
            current (Point3D):
                current position [x, y, z]

        Returns:
            list[float]:
                single line a data to be saved in a CSV file
        """
        
        goal_direction: Direction3D = self.compute_goal_direction(current)

        nearest_obstacle_distance: float = (self.compute_nearest_obstacle_distance(current))

        row: list[float] = [
            current[0],
            current[1],
            current[2],
            self.goal[0],
            self.goal[1],
            self.goal[2],
            nearest_obstacle_distance,
            goal_direction[0],
            goal_direction[1],
            goal_direction[2],
        ]
        
        return row
    
    
    def generate(self, num_samples: int = 10000) -> None:
        """
        Generate sampling dataset and save CSV file
        
        Args:
            num_samples (int):
                number of sample
                
        Returns:
            None
        """
        
        header: list[str] = [
            'current_x',
            'current_y',
            'current_z',
            'goal_x',
            'goal_y',
            'goal_z',
            'nearest_obstacle_distance',
            'target_dx',
            'target_dy',
            'target_dz',
        ]
        
        with open(self.dataset_path, 'w', newline= "") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)

            for sample_index in  range(num_samples):
                current: Point3D = self.sample_valid_point()
                
                row: list[float] = self.create_dataset_row(current)

                writer.writerow(row)

                if sample_index % 1000 == 0:
                    print(f'Generated {sample_index} / {num_samples} samples')

        print(f'Dataset saved to : {self.dataset_path}')

        
def main() -> None:
    generator = SamplingDatasetGenerator()
    generator.generate(num_samples= 10000)

if __name__ == "__main__":
    main()
