from neural_rrt_planner.planners.informed_rrt_star import InformedRRTStar


def main():
    planner = InformedRRTStar()

    path = planner.plan()

    print("\n===== Informed RRT* Result =====")

    if not path:
        print("No path found")
        return

    print(f"Node count: {len(planner.nodes)}")
    print(f"Path cost: {planner.best_goal_node.cost:.3f}")
    print(f"Waypoint count: {len(path)}")

    for i, waypoint in enumerate(path):
        print(f"{i}: {waypoint}")


if __name__ == "__main__":
    main()