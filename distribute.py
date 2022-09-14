
import random
import time
import networkx as nx
import numpy as np

# 不考虑负载，采用最短路径（最少跳数）
def NFsDistribution_K_means_load_dalay(Sat_load, amf_num, smf_num, umf_num, Delay_Matrix, orbitNum, satsNum):
    G_networkx = nx.Graph()
    for i in range(len(Delay_Matrix)):
        for j in range(len(Delay_Matrix)):
            if Delay_Matrix[i][j] != 0:
                G_networkx.add_edge(i, j, weight=float(1 / Delay_Matrix[i][j]))

    AMF_location = np.zeros([satsNum, orbitNum])
    SMF_location = np.zeros([satsNum, orbitNum])
    UPF_location = np.zeros([satsNum, orbitNum])
    amf = []
    i = 0
    while True:
        a = random.randint(0, orbitNum-1)
        b = random.randint(0, satsNum-1)
        if [a, b] not in amf:
            amf.append([a, b])
            i = i + 1
            if i == amf_num:
                break
    print("amf", amf)
    while True:
        ok, amf, Cluster = K_means_load_delay(amf, amf_num, G_networkx,satsNum,orbitNum,Sat_load)
        if ok == 0:
            break

    print("amf", amf)
    # a = 0
    # for i in range(amf_num):
    #     center = []
    #     for m in range(len(Cluster[i])):
    #         l = []
    #         for j in Cluster[i]:
    #             l.append(Sat_load[j[1]][j[0]]*nx.dijkstra_path_length(G_networkx, j[0] * satsNum + j[1],
    #                                              Cluster[i][m][0] * satsNum + Cluster[i][m][1]))
    #         center.append(sum(l))
    #     a = a + min(center)
    # print(a)
    for i in range(amf_num):
        AMF_location[amf[i][1]][amf[i][0]] = 1

    SMF_location = AMF_location
    UPF_location = AMF_location
    return AMF_location, SMF_location, UPF_location

def  K_means_load_delay(amf,K,G_networkx,satsNum,orbitNum,Sat_load):
    Cluster = [[] for _ in range(K)]
    for i in range(orbitNum):
        for j in range(satsNum):
            l = []
            for m in range(K):
                l.append(nx.dijkstra_path_length(G_networkx, amf[m][0] * satsNum + amf[m][1], satsNum * i + j))
            Cluster[l.index(min(l))].append([i, j])
    # print(Cluster)
    center = []
    amf_new = []
    for i in range(K):
        center = []
        for m in range(len(Cluster[i])):
            l = []
            for j in Cluster[i]:
                l.append(Sat_load[j[1]][j[0]] * nx.dijkstra_path_length(G_networkx, j[0] * satsNum + j[1],
                                                 Cluster[i][m][0] * satsNum + Cluster[i][m][1]))
            center.append(sum(l))
        amf_new.append(Cluster[i][center.index(min(center))])
    # print("Cluster", Cluster)
    print("amf_new",amf_new)

    k = 0
    for i in range(len(amf)):
        if nx.shortest_path_length(G_networkx, amf[i][0] * satsNum + amf[i][1], amf_new[i][0] * satsNum + amf_new[i][1]) > 0:
            amf[i] = amf_new[i]
            k = 1
    if k == 0:
        return 0,amf,Cluster
    else:
        return 1, amf, Cluster


# 不考虑负载，采用最短路径（最少跳数）
def NFsDistribution_K_means(Sat_load, amf_num, smf_num, umf_num, Adjacency_Matrix, orbitNum, satsNum):
    G_networkx = nx.Graph()
    for i in range(len(Adjacency_Matrix)):
        for j in range(len(Adjacency_Matrix)):
            if Adjacency_Matrix[i][j] != 0:
                # G_networkx.add_edge(i, j, weight=float(1 / Adjacency_Matrix[i][j]))
                G_networkx.add_edge(i, j)

    AMF_location = np.zeros([satsNum, orbitNum])
    SMF_location = np.zeros([satsNum, orbitNum])
    UPF_location = np.zeros([satsNum, orbitNum])
    amf = []
    i = 0
    while True:
        a = random.randint(0, orbitNum)
        b = random.randint(0, satsNum)
        if [a, b] not in amf:
            amf.append([a, b])
            i = i + 1
            if i == amf_num:
                break

    while True:
        ok, amf, Cluster = K_means_hop(amf, amf_num, G_networkx,satsNum,orbitNum)
        if ok == 0:
            break

    # print("amf", amf)
    # a = 0
    # for i in range(amf_num):
    #     center = []
    #     for m in range(len(Cluster[i])):
    #         l = []
    #         for j in Cluster[i]:
    #             l.append(nx.shortest_path_length(G_networkx, j[0] * satsNum + j[1],
    #                                              Cluster[i][m][0] * satsNum + Cluster[i][m][1]))
    #         center.append(sum(l))
    #     a = a + min(center)
    # print(a)
    for i in range(amf_num):
        AMF_location[amf[i][1]][amf[i][0]] = 1

    SMF_location = AMF_location
    UPF_location = AMF_location
    return AMF_location, SMF_location, UPF_location

def  K_means_hop(amf,K,G_networkx,satsNum,orbitNum):
    sat_cluster = []
    Cluster = [[] for _ in range(K)]
    for i in range(orbitNum):
        for j in range(satsNum):
            l = []
            for m in range(K):
                l.append(nx.shortest_path_length(G_networkx, amf[m][0] * satsNum + amf[m][1], satsNum * i + j))
            sat_cluster.append(l.index(min(l)))
            Cluster[l.index(min(l))].append([i, j])
    # print(sat_cluster)
    # print(Cluster)
    center = []
    amf_new = []
    for i in range(K):
        center = []
        for m in range(len(Cluster[i])):
            l = []
            for j in Cluster[i]:
                l.append(nx.shortest_path_length(G_networkx, j[0] * satsNum + j[1],
                                                 Cluster[i][m][0] * satsNum + Cluster[i][m][1]))
            center.append(sum(l))
        amf_new.append(Cluster[i][center.index(min(center))])
    # print("Cluster", Cluster)
    # print("amf_new",amf_new)

    k = 0
    for i in range(len(amf)):
        if nx.shortest_path_length(G_networkx, amf[i][0] * satsNum + amf[i][1], amf_new[i][0] * satsNum + amf_new[i][1]) > 0:
            amf[i] = amf_new[i]
            k = 1
    if k == 0:
        return 0,amf,Cluster
    else:
        return 1, amf, Cluster




def NFsDistribution_Suiji(Sat_load,amf_num,smf_num,upf_num,satsNum,orbitNum):
    AMF_location = np.zeros([satsNum, orbitNum])
    # AMF_location[[0], :] = 1
    i = 0
    while True:
        a = random.randint(0, 9)
        b = random.randint(0, 7)
        if AMF_location[a][b] != 1:
            AMF_location[a][b] = 1
            i = i + 1
            if i == amf_num:
                break

    SMF_location = np.zeros([satsNum, orbitNum])
    i = 0
    while True:
        a = random.randint(0, 9)
        b = random.randint(0, 7)
        if SMF_location[a][b] != 1:
            SMF_location[a][b] = 1
            i = i + 1
            if i == smf_num:
                break

    UPF_location = np.zeros([satsNum, orbitNum])
    # UPF_location[[2], :] = 1
    # print(AMF_location)
    i = 0
    while True:
        a = random.randint(0, 9)
        b = random.randint(0, 7)
        if UPF_location[a][b] != 1:
            UPF_location[a][b] = 1
            i = i + 1
            if i == upf_num:
                break
    return AMF_location,SMF_location,UPF_location

def NFsDistribution_Guding(Sat_load,satsNum,orbitNum):
    AMF_location = np.zeros([satsNum, orbitNum])
    AMF_location[[0], :] = 1
    # AMF_location[[0], 0:] = 1

    SMF_location = np.zeros([satsNum, orbitNum])
    SMF_location[[1], :] = 1

    UPF_location = np.zeros([satsNum, orbitNum])
    UPF_location[[2], :] = 1

    return AMF_location,SMF_location,UPF_location
