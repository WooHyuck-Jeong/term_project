import os
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob

dataDir = "/Users/woohyuck/Documents/02_workspace/04_Lecture/01_Mobility/MyProject/term_project/benchmark"
dataList : list[str] = glob(os.path.join(dataDir, "*.csv"))

# for i in range(len(dataList)):
#     print(i)
#     print(dataList[i])

rawDataIdx = [2, 4, 0, 1]
rawDataList = [dataList[i] for i in rawDataIdx]

rawData : list[pd.DataFrame] = [pd.read_csv(rawDataList[i]) for i in range(len(rawDataList))]

# for i in range(len(rawData)):
#     print(i)
#     print(rawData[i])

"""
0: rrt*
1: informed-rrt*
2: neural-rrt* v1
3: neural-rrt* v2
"""

rrt_star = rawData[0].copy()
informed_rrt_star = rawData[1].copy()
neural_rrt_v1 = rawData[2].copy()
neural_rrt_v2 = rawData[3].copy()

# print(rrt_star.columns)


fig, ax = plt.subplots(4, 1, figsize= (15, 10))
# plot first_path_iter
rrt_star.plot(kind= "line", x= "run", y= "first_path_iter", title= "First Path Iter", ax= ax[0], marker= "+", c= 'red', label= 'rrt*', ylim= [0, 500], grid= True)
informed_rrt_star.plot(kind= "line", x= "run", y= "first_path_iter", title= "First Path Iter", ax= ax[0], marker= "^", c= 'blue', label= 'Informed rrt*', ylim= [0, 500], grid= True)
neural_rrt_v1.plot(kind= "line", x= "run", y= "first_path_iter", title= "First Path Iter", ax= ax[0], marker= "d", c= 'green', label= 'Neural-rrt* v1', ylim= [0, 500], grid= True)
neural_rrt_v2.plot(kind= "line", x= "run", y= "first_path_iter", title= "First Path Iter", ax= ax[0], marker= "o", c= 'orange', label= 'Neural-rrt* v2', ylim= [0, 500], grid= True)

# plot final cost
rrt_star.plot(kind= "line", x= "run", y= "final_cost", title= "Final Cost", ax= ax[1], marker= "+", c= 'red', label= 'rrt*', grid= True)
informed_rrt_star.plot(kind= "line", x= "run", y= "final_cost", title= "Final Cost", ax= ax[1], marker= "^", c= 'blue', label= 'Informed rrt*', grid= True)
neural_rrt_v1.plot(kind= "line", x= "run", y= "final_cost", title= "Final Cost", ax= ax[1], marker= "d", c= 'green', label= 'Neural-rrt* v1', grid= True)
neural_rrt_v2.plot(kind= "line", x= "run", y= "final_cost", title= "Final Cost", ax= ax[1], marker= "o", c= 'orange', label= 'Neural-rrt* v2', grid= True)

# plot final nodes
rrt_star.plot(kind= "line", x= "run", y= "final_nodes", title= "Final Nodes", ax= ax[2], marker= "+", c= 'red', label= 'rrt*', ylim= [1700, 2100], grid= True)
informed_rrt_star.plot(kind= "line", x= "run", y= "final_nodes", title= "Final Nodes", ax= ax[2], marker= "^", c= 'blue', label= 'Informed rrt*', ylim= [1700, 2100], grid= True)
neural_rrt_v1.plot(kind= "line", x= "run", y= "final_nodes", title= "Final Nodes", ax= ax[2], marker= "d", c= 'green', label= 'Neural-rrt* v1', ylim= [1700, 2100], grid= True)
neural_rrt_v2.plot(kind= "line", x= "run", y= "final_nodes", title= "Final Nodes", ax= ax[2], marker= "o", c= 'orange', label= 'Neural-rrt* v2', ylim= [1700, 2100], grid= True)

# plot elapsed time
rrt_star.plot(kind= "line", x= "run", y= "elapsed_sec", title= "Elapsed time[s]", ax= ax[3], marker= "+", c= 'red', label= 'rrt*', grid= True)
informed_rrt_star.plot(kind= "line", x= "run", y= "elapsed_sec", title= "Elapsed time[s]", ax= ax[3], marker= "^", c= 'blue', label= 'Informed rrt*', grid= True)
neural_rrt_v1.plot(kind= "line", x= "run", y= "elapsed_sec", title= "Elapsed time[s]", ax= ax[3], marker= "d", c= 'green', label= 'Neural-rrt* v1', grid= True)
neural_rrt_v2.plot(kind= "line", x= "run", y= "elapsed_sec", title= "Elapsed time[s]", ax= ax[3], marker= "o", c= 'orange', label= 'Neural-rrt* v2', grid= True)

for i in range(4):
    ax[i].legend(loc= 'upper right')
plt.tight_layout()
plt.show()

# fig.savefig("Benchmark_Compare_v2.png", dpi= 500)
# plt.close()