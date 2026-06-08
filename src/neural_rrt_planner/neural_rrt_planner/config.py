"""
Configuration file for AUV RRT* path planning.

Coordinate convention:
- x: tank length direction [m]
- y: tank width direction [m]
- z: depth direction [m]
- water surface: z = 0.0
- downward depth: negative z
"""

MAP_CONFIG = {
    # 실제 회류수조 크기 기반 map boundary
    "bounds": {
        "x": [0.0, 5.0],
        "y": [0.0, 2.0],
        "z": [-1.6, 0.0],
    },

    # 시작점과 목표점
    # 실제 실험에서는 AUV 투입 위치를 start 기준으로 맞추면 됨
    "start": [0.4, 0.4, -0.5],
    "goal": [4.6, 1.6, -1.0],

    # 원기둥 장애물
    # center: [x, y, z]
    # radius: cylinder radius [m]
    # height: cylinder height [m]
    "obstacles": [
        {
            "type": "cylinder",
            "name": "obstacle_1",
            "center": [1.7, 1.0, -0.8],
            "radius": 0.18,
            "height": 1.2,
        },
        {
            "type": "cylinder",
            "name": "obstacle_2",
            "center": [3.0, 0.7, -0.8],
            "radius": 0.20,
            "height": 1.2,
        },
        {
            "type": "cylinder",
            "name": "obstacle_3",
            "center": [3.8, 1.4, -0.8],
            "radius": 0.16,
            "height": 1.2,
        },
    ],

    # 장애물 및 벽과의 안전 거리
    "safety_margin": 0.10,

    # 실제 planning에 사용할 허용 수심 범위
    # 수면과 바닥 충돌 방지를 위해 실제 tank boundary보다 좁게 설정
    "depth_limit": {
        "z_min": -1.4,
        "z_max": -0.2,
    },

    # RRT* 공통 파라미터
    "rrt_params": {
        "step_size": 0.20,
        "goal_sample_rate": 0.08,
        "max_iter": 2000,
        "search_radius": 0.45,
        "goal_threshold": 0.25,
        "collision_check_resolution": 0.03,
    },

    # Informed RRT* 파라미터
    "informed_params": {
        "enable_ellipsoid_visualization": True,
    },

    # Neural-guided RRT* 파라미터
    "neural_params": {
        "neural_sample_rate": 0.80,
        "random_sample_rate": 0.20,
        "sample_distance": 0.35,
        "model_path": "models/sampling_mlp.pth",
    },

    # RViz2 시각화 설정
    "visualization": {
        "frame_id": "map",
        "tree_topic": "/neural_rrt/tree",
        "path_topic": "/neural_rrt/path",
        "marker_topic": "/neural_rrt/markers",
    },
}