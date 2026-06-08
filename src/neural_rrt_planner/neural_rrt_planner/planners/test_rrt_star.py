from neural_rrt_planner.planners.rrt_star import RRTStar

def main():
    planner = RRTStar()
    nodes = planner.plan()
    
    print('Generated node count: ', len(nodes))

    print('First node: ', nodes[0])
    print('Last node: ', nodes[-1])

if __name__ == "__main__":
    main()