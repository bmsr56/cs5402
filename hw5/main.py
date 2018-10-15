import sys
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import csv
import copy

def getDF(csvFile):
    table = pd.read_csv(csvFile, sep= ',')
    return table

def findManhattan(p1, p2, c1, c2):
    totaldist = abs(p1 - c1) - abs(p2 - c2)
    return totaldist

def findKmeans(dataList, centroidL, distMetric, iter):
    s = True
    centroidList = copy.deepcopy(centroidL)
    while(s):
        centroidGroups = {tuple(centroidList[0]): [], tuple(centroidList[1]): []}
        for pair in dataList:
            distances = []
            for centroid in centroidList:
                if distMetric == 'm':
                    distances.append(
                        findManhattan(
                            pair[0], pair[1],
                            centroid[0], centroid[1])        
                        )
            if distances[0] < distances[1]:
                centroidGroups[centroidList[0]].append(pair)
            else:
                centroidGroups[centroidList[1]].append(pair)

        newCentroidList = updateCentroids(centroidGroups)
        for i in newCentroidList:
            if i in centroidList:
                s = False
    return newCentroidList

def updateCentroids(centroidGroups):
    centroidList = []
    for centroid, pairs in centroidGroups.items():
        length = len(pairs)
        x1_tot = 0
        x2_tot = 0
        c1 = 0
        c2 = 0
        for i in pairs:
            x1_tot += pairs[0]
            x2_tot += pairs[0]
        c1 = x1_tot/length
        c2 = x2_tot/length
        centroidList.append([c1, c2])
    return centroidList

def main():
    
    # x1 = [3,3,2,2,6,6,7,7,8,7]
    # x2 = [5,4,8,3,2,4,3,4,5,6]
    # dataList = []
    # for i, j in zip(x1, x2):
    #     dataList.append([x for x in [i, j]])
    # print(dataList)

    # print(findKmeans(dataList, [[4, 6],[5, 4]], 'm', 1))
    # return
    df = getDF(sys.argv[1])
    df_copy = df
    # df_dum = pd.get_dummies(df_copy, columns=['team'])
    # columns = ['team_X1','team_X2','team_X3','team_X4',
    #             'team_X5','team_X6','team_X7','team_X8',
    #             'team_X9','team_X10','wins2016','wins2017']

    print(df_copy)

    # # Question 1_1
    centroid = np.array([[3,2], [4,8]], np.float64)
    kmeans = KMeans(n_clusters=2, init=centroid, n_init=1).fit(df_copy)
    labels = kmeans.labels_
    # df_copy['clusters'] = labels
    # df_copy.extend(['clusters'])

    # results = df_copy[columns].groupby(['clusters']).mean()
    print(kmeans.cluster_centers_)


    # # Question 1_2
    # # Question 1_3
    # # Question 1_4

if __name__ == '__main__':
    main()