import time
import random
import networkx as nx
import numpy as np
import nsga2
import matplotlib.pyplot as plt
import basic_line
from line_profiler import LineProfiler


def CreateNetworkX(G_networkx, G_Matrix):
    for i in range(len(G_Matrix)):
        for j in range(len(G_Matrix)):
            if G_Matrix[i][j] >= 1:
                G_networkx.add_edge(i, j, weight=1)
    return G_networkx


def CreateNetworkX_delay(G_networkx, G_Matrix):
    for i in range(len(G_Matrix)):
        for j in range(len(G_Matrix)):
            if G_Matrix[i][j] >= 1:
                G_networkx.add_edge(i, j, weight=2.27 + random.random() / 10)
    return G_networkx


def CreateNetworkX_test(G_networkx, G_Matrix):
    for i in range(len(G_Matrix)):
        for j in range(len(G_Matrix)):
            if G_Matrix[i][j] > 0:
                G_networkx.add_edge(i, j, weight=1000 / (G_Matrix[i][j] + random.random() / 10))
    return G_networkx


def create_grid(raw=3, column=2, MAXRAW=100, MAXCOL=100):
    count = 0
    map = np.zeros([MAXRAW * MAXCOL, MAXRAW * MAXCOL])

    for j in range(1, raw + 1):
        count = 0
        while count < column:
            map[j + count * raw][j + count * raw + raw] = 1
            count = count + 1
    # print(map)
    i = 1
    for count in range(column):
        i = count * raw + 1
        for j in range(1, raw):
            map[i + j - 1][i + j] = 1
    count = 0
    while count < column:
        map[1 + count * raw][1 + count * raw + raw - 1] = 1
        count = count + 1
    # for i_1 in range(1, raw * column):
    #     for i_2 in range(1, raw * column):
    # print(map[i_1][i_2])
    # print("\n")
    # print("-----")
    # print(map)
    # print(map[1:raw*column+1])
    a = map[1:raw * column + 1, 1:raw * column + 1]
    for i in range(len(a)):
        for j in range(i, len(a)):
            a[j][i] = a[i][j]
    return a


def AMF_ProcessTime(current_resource):
    max_resource = 1
    min_resource = 0.2
    max_process_time = 4  # ms
    min_process_time = 0.1
    a = (max_process_time - min_process_time) / (min_resource - max_resource)
    b = (min_resource * min_process_time - max_process_time * max_resource) / (min_resource - max_resource)
    y = a * current_resource + b
    return y


def SMF_ProcessTime(current_resource):
    max_resource = 1
    min_resource = 0.2
    max_process_time = 6  # ms
    min_process_time = 0.4
    a = (max_process_time - min_process_time) / (min_resource - max_resource)
    b = (min_resource * min_process_time - max_process_time * max_resource) / (min_resource - max_resource)
    y = a * current_resource + b
    return y


def UPF_ProcessTime(current_resource):
    max_resource = 1
    min_resource = 0.2
    max_process_time = 5  # ms
    min_process_time = 0.2
    a = (max_process_time - min_process_time) / (min_resource - max_resource)
    b = (min_resource * min_process_time - max_process_time * max_resource) / (min_resource - max_resource)
    y = a * current_resource + b
    return y


def TransTime_PTime(LoadSat, delay, AMF_NUMBER, SMF_NUMBER, UPF_NUMBER, amf_location, smf_location, upf_location,
                    resource_sat):
    # 单位都是ms
    # 链路的时间都是 1ms
    delay_sat_vnf1_var = np.matmul(amf_location, delay)
    delay_sat_vnf2_var = np.matmul(smf_location, delay)
    delay_vnf1_vnf2_var = np.matmul(smf_location, delay_sat_vnf1_var.T)
    delay_vnf2_vnf3_var = np.matmul(upf_location, delay_sat_vnf2_var.T)

    # delay_sat_vnf1_min_var = np.min(delay_sat_vnf1_var, axis=0)
    vnf1_index_var = np.argmin(delay_sat_vnf1_var, axis=0)
    # delay_vnf1_vnf2__min_var = np.min(delay_vnf1_vnf2_var, axis=0)
    vnf2_index_var = np.argmin(delay_vnf1_vnf2_var, axis=0)
    # delay_vnf2_vnf3__min_var = np.min(delay_vnf2_vnf3_var, axis=0)
    vnf3_index_var = np.argmin(delay_vnf2_vnf3_var, axis=0)
    # 每个VNF部署节点
    vnf1_location_var = np.argmax(amf_location, 1)
    vnf2_location_var = np.argmax(smf_location, 1)
    vnf3_location_var = np.argmax(upf_location, 1)
    # print('vnf1_location_var', vnf1_location_var)
    sat_link_index = np.zeros((delay.shape[0], 3)).astype(np.uint8)
    # 消息路径
    for s in range(delay.shape[0]):
        sat_link_index[s][0] = vnf1_location_var[vnf1_index_var[s]]
        sat_link_index[s][1] = vnf2_location_var[vnf2_index_var[vnf1_index_var[s]]]
        sat_link_index[s][2] = vnf3_location_var[vnf3_index_var[vnf2_index_var[vnf1_index_var[s]]]]
    # print('sat_link_index', sat_link_index)
    trass_time = np.zeros((delay.shape[0], 5))
    for s in range(delay.shape[0]):
        trass_time[s][0] = delay[s, sat_link_index[s][0]]
        trass_time[s][1] = delay[sat_link_index[s][0], sat_link_index[s][1]]
        trass_time[s][2] = delay[sat_link_index[s][1], sat_link_index[s][2]]
    business_3 = np.zeros((delay.shape[0]))
    business_4 = np.zeros((delay.shape[0]))
    business_5 = np.zeros((delay.shape[0]))
    for s in range(delay.shape[0]):
        business_3[s] = 2 * trass_time[s][0] + 2 * trass_time[s][1] + 2 * trass_time[s][2] + 4 * AMF_ProcessTime(
            resource_sat[sat_link_index[s][0]]) + \
                        4 * SMF_ProcessTime(resource_sat[sat_link_index[s][1]]) + 2 * UPF_ProcessTime(
            resource_sat[sat_link_index[s][2]])
        business_4[s] = 7 * trass_time[s][0] + 13 * AMF_ProcessTime(resource_sat[sat_link_index[s][0]]) + 6 * \
                        trass_time[s][1] + \
                        10 * SMF_ProcessTime(resource_sat[sat_link_index[s][1]]) + 4 * trass_time[s][
                            2] + 4 * UPF_ProcessTime(resource_sat[sat_link_index[s][2]])
        business_5[s] = 3 * trass_time[s][0] + 2 * trass_time[s][1] + 2 * trass_time[s][2] + 5 * AMF_ProcessTime(
            resource_sat[sat_link_index[s][0]]) + \
                        4 * SMF_ProcessTime(resource_sat[sat_link_index[s][1]]) + 2 * UPF_ProcessTime(
            resource_sat[sat_link_index[s][2]])
        # print('process_time', process_time)
    Z = 0
    for s in range(delay.shape[0]):
        if ground_CN:
            Z += LoadSat[s] * (0.6 * business_3[s] + 0.3 * business_4[s] + 0.1 * business_5[s])
        else:
            Z += LoadSat[s] * (0.8 * business_3[s] + 0.1 * business_4[s] + 0.1 * business_5[s])
    return Z


def CAPITAL(COST_AMF, COST_SMF, COST_UPF, AMF_NUMBER, SMF_NUMBER, UPF_NUMBER):
    return COST_AMF * AMF_NUMBER + COST_SMF * SMF_NUMBER + COST_UPF * UPF_NUMBER


def LinkCost(LoadSat, delay, hop, LINK_COST_3, LINK_COST_4, LINK_COST_5, amf_location, smf_location,
             upf_location):
    delay_sat_vnf1_var = np.matmul(amf_location, delay)
    delay_sat_vnf2_var = np.matmul(smf_location, delay)
    delay_vnf1_vnf2_var = np.matmul(smf_location, delay_sat_vnf1_var.T)
    delay_vnf2_vnf3_var = np.matmul(upf_location, delay_sat_vnf2_var.T)

    # delay_sat_vnf1_min_var = np.min(delay_sat_vnf1_var, axis=0)
    vnf1_index_var = np.argmin(delay_sat_vnf1_var, axis=0)
    # delay_vnf1_vnf2__min_var = np.min(delay_vnf1_vnf2_var, axis=0)
    vnf2_index_var = np.argmin(delay_vnf1_vnf2_var, axis=0)
    # delay_vnf2_vnf3__min_var = np.min(delay_vnf2_vnf3_var, axis=0)
    vnf3_index_var = np.argmin(delay_vnf2_vnf3_var, axis=0)

    hop_sat_vnf1_var = np.matmul(amf_location, hop)
    hop_sat_vnf2_var = np.matmul(smf_location, hop)
    hop_vnf1_vnf2_var = np.matmul(smf_location, hop_sat_vnf1_var.T)
    hop_vnf2_vnf3_var = np.matmul(upf_location, hop_sat_vnf2_var.T)
    # 每个VNF部署节点
    # vnf1_location_var = np.argmax(amf_location, 1)
    # vnf2_location_var = np.argmax(smf_location, 1)
    # vnf3_location_var = np.argmax(upf_location, 1)
    # print('vnf1_location_var', vnf1_location_var)

    sat_link_hop = np.zeros((delay.shape[0], 5))
    # 0 sat-amf 1 amf-smf  2 smf-upf 3 amf-ground 4 smf-ground
    for i in range(delay.shape[0]):
        sat_link_hop[i][0] = hop_sat_vnf1_var[vnf1_index_var[i]][i]
        sat_link_hop[i][1] = hop_vnf1_vnf2_var[vnf2_index_var[vnf1_index_var[i]]][vnf1_index_var[i]]
        sat_link_hop[i][2] = hop_vnf2_vnf3_var[vnf3_index_var[vnf2_index_var[vnf1_index_var[i]]]][
            vnf2_index_var[vnf1_index_var[i]]]

    business_3 = np.zeros((delay.shape[0]))
    business_4 = np.zeros((delay.shape[0]))
    business_5 = np.zeros((delay.shape[0]))
    for i in range(delay.shape[0]):
        business_3[i] = LINK_COST_3[0] * sat_link_hop[i][0] + \
                        LINK_COST_3[1] * sat_link_hop[i][1] + \
                        LINK_COST_3[2] * sat_link_hop[i][2] + \
                        LINK_COST_3[3] * sat_link_hop[i][2] + \
                        LINK_COST_3[4] * sat_link_hop[i][1] + \
                        LINK_COST_3[5] * sat_link_hop[i][0]
        business_4[i] = LINK_COST_4[0] * sat_link_hop[i][0] + \
                        LINK_COST_4[1] * sat_link_hop[i][1] + \
                        LINK_COST_4[2] * sat_link_hop[i][1] + \
                        LINK_COST_4[3] * sat_link_hop[i][0] + \
                        LINK_COST_4[4] * sat_link_hop[i][0] + \
                        LINK_COST_4[5] * sat_link_hop[i][1] + \
                        LINK_COST_4[6] * sat_link_hop[i][2] + \
                        LINK_COST_4[7] * sat_link_hop[i][2] + \
                        LINK_COST_4[8] * sat_link_hop[i][1] + \
                        LINK_COST_4[9] * sat_link_hop[i][0] + \
                        LINK_COST_4[10] * sat_link_hop[i][0] + \
                        LINK_COST_4[11] * sat_link_hop[i][1] + \
                        LINK_COST_4[12] * sat_link_hop[i][2] + \
                        LINK_COST_4[13] * sat_link_hop[i][2] + \
                        LINK_COST_4[14] * sat_link_hop[i][1] + \
                        LINK_COST_4[15] * sat_link_hop[i][0] + \
                        LINK_COST_4[16] * sat_link_hop[i][0]
        business_5[i] = LINK_COST_5[0] * sat_link_hop[i][0] + \
                        LINK_COST_5[1] * sat_link_hop[i][1] + \
                        LINK_COST_5[2] * sat_link_hop[i][2] + \
                        LINK_COST_5[3] * sat_link_hop[i][2] + \
                        LINK_COST_5[4] * sat_link_hop[i][1] + \
                        LINK_COST_5[5] * sat_link_hop[i][0] + \
                        LINK_COST_5[6] * sat_link_hop[i][0]

    Z = 0
    for s in range(delay.shape[0]):
        if ground_CN:
            Z += LoadSat[s] * (0.6 * business_3[s] + 0.3 * business_4[s] + 0.1 * business_5[s])
        else:
            Z += LoadSat[s] * (0.8 * business_3[s] + 0.1 * business_4[s] + 0.1 * business_5[s])
    reliability_3 = np.zeros((delay.shape[0]))
    reliability_4 = np.zeros((delay.shape[0]))
    reliability_5 = np.zeros((delay.shape[0]))
    R = 0
    for s in range(delay.shape[0]):
        reliability_3[s] = 0.999 ** (2 * sat_link_hop[s][0] + 2 * sat_link_hop[s][1] + 2 * sat_link_hop[s][2])
        reliability_4[s] = 0.999 ** (7 * sat_link_hop[s][0] + 6 * sat_link_hop[s][1] + 4 * sat_link_hop[s][2])
        reliability_5[s] = 0.999 ** (3 * sat_link_hop[s][0] + 2 * sat_link_hop[s][1] + 2 * sat_link_hop[s][2])

    for s in range(delay.shape[0]):
        if ground_CN:
            R += LoadSat[s] * (0.6 * reliability_3[s] + 0.3 * reliability_4[s] + 0.1 * reliability_5[s])
        else:
            R += LoadSat[s] * (0.8 * reliability_3[s] + 0.1 * reliability_4[s] + 0.1 * reliability_5[s])
    return Z, R


def F(AMF_NUMBER, SMF_NUMBER, UPF_NUMBER, amf_location, smf_location,
      upf_location):
    capital = CAPITAL(COST_AMF, COST_SMF, COST_UPF, AMF_NUMBER, SMF_NUMBER, UPF_NUMBER)
    tptime = TransTime_PTime(LoadSat, delay, AMF_NUMBER, SMF_NUMBER, UPF_NUMBER, amf_location, smf_location,
                             upf_location,
                             resource_sat)
    linkcost, reliability = LinkCost(LoadSat, delay, hop, LINK_COST_3, LINK_COST_4, LINK_COST_5,
                                     amf_location, smf_location, upf_location)
    opl = _mu * capital + _nu * tptime + _lambda * linkcost - zeta * reliability
    # print('capital', capital, 'time', tptime, 'linkcost', linkcost,'reliability ',reliability,'\nopl', opl)
    # qqq.append(tptime)
    # qq2.append(linkcost)
    # qq1.append(reliability)
    # print(min(qqq),max(qqq),min(qq2),max(qq2),min(qq1),max(qq1))
    # print( _mu * capital, _nu * time + _lambda * (linkcost - zeta * reliability))
    return _mu * capital, _nu * tptime + _lambda * linkcost - zeta * reliability, opl, capital, tptime, linkcost, reliability


def translateDNA(l_amf, l_smf, l_upf):
    amf = np.zeros((len(l_amf), 960))
    smf = np.zeros((len(l_smf), 960))
    upf = np.zeros((len(l_upf), 960))
    for i in range(len(l_amf)):
        amf[i][int(l_amf[i])] = 1
    for i in range(len(l_smf)):
        smf[i][int(l_smf[i])] = 1
    for i in range(len(l_upf)):
        upf[i][int(l_upf[i])] = 1
    return amf, smf, upf


compute_nsga2 = False
Test_nsga2 = True
ground_CN = True
qqq = []
qq2 = []
qq1 = []
# 业务3 用户XN切换 amf smf upf 0.7
LINK_COST_3 = [160, 554 , 142, 73, 294, 139]
# 业务4 用户N2切换 amf 0.2
LINK_COST_4 = [1630, 374, 191, 1786, 678, 381, 142, 73, 167, 658, 118, 352, 156, 73, 100, 102, 126]
# 业务5 业务请求 amf 0.2
LINK_COST_5 = [166, 365, 142, 73, 167, 1262, 118]
COST_AMF = 300
COST_SMF = 200
COST_UPF = 400
# resource_sat = np.array([1 for i in range(960)])
if ground_CN:
    resource_sat = [1 for _ in range(960)]
else:
    resource_sat = np.loadtxt('./resource.txt')
_mu = 1
_nu = 39.8
_lambda = 0.38
zeta = 83665
# _mu = 16
# _nu = 105
# _lambda = 6.3
# zeta = 52631
if compute_nsga2:
    LoadSat = np.loadtxt('./loadsum.txt').T.reshape(1, -1)[0]
    a = sum(LoadSat)
    LoadSat = [x / a for x in LoadSat]
    # leo = create_grid(40, 24)
    # leo = np.loadtxt('./data/0Adjacency_Matrix.txt')[:960, :960]
    # G = nx.Graph()
    # G = CreateNetworkX_delay(G, leo)
    # delay = np.zeros((960, 960))
    # for i in range(960):
    #     for j in range(960):
    #         delay[i][j] = nx.dijkstra_path_length(G, i, j)
    delay = np.loadtxt('./delay.txt')
    # # x = create_grid(40, 24)
    # G_hop = nx.Graph()
    # G_hop = CreateNetworkX(G_hop, leo)
    # nx.draw(G_hop, with_labels=True)
    # plt.show()
    # hop = np.zeros((960, 960))
    # for i in range(960):
    #     for j in range(960):
    #         hop[i][j] = nx.shortest_path_length(G_hop, i, j)\
    hop = np.loadtxt('./hop.txt')

if Test_nsga2:

    tptime = []
    linkcost = []
    reliability = []
    capital = []
    # amf = [113, 860, 685, 682, 200, 207, 202, 873, 587, 189, 192, 100,  98, 999,  77]
    # smf = [191, 859, 135, 683,  97, 206, 203, 872, 586,  74, 999,  37, 999, 999, 999]
    # upf = [190, 858, 585, 684,  96, 205,  75, 871, 999, 999, 999, 999, 999, 999, 999]
    # amf =[76, 824, 508, 812, 123, 133, 764, 104, 272, 340, 577, 408, 478, 834]
    # smf =[122, 833, 507, 852, 368, 93, 724, 144, 477, 300, 578]
    # upf =[121, 832, 476, 853, 328, 53, 684, 184]
    # amf = list(filter(lambda x: x != 999, amf))
    # smf = list(filter(lambda x: x != 999, smf))
    # upf = list(filter(lambda x: x != 999, upf))
    # print(amf, len(amf),smf,len(smf), upf,len(upf))
    # amf, smf, upf = translateDNA(amf, smf, upf)
    for T in range(420,520):
        print('T', T,'ground_CN',ground_CN)
        amf = list(np.loadtxt('./delayhop5/data/' + str(T) + 'AMF_location.txt'))
        print(amf)
        smf = amf
        upf = amf
        amf, smf, upf = translateDNA(amf, smf, upf)
        # LoadSat = list(np.loadtxt('./data/' + str(T) + 'Sat_load.txt').T.reshape(1, -1)[0])
        LoadSat = list(np.loadtxt('./delayhop5/data/' + str(T) + 'Sat_load.txt').T.reshape(1, -1)[0])
        a = sum(LoadSat)
        LoadSat = [x / a for x in LoadSat]
        delay = np.loadtxt('./delayhop5/result/' + str(T) + 'delay.txt')
        hop = np.loadtxt('./delayhop5/result/' + str(T) + 'hop.txt')
        # x = np.loadtxt('./data/' + str(T) + 'Delay_Matrix.txt')
        # G = nx.Graph()
        # G = CreateNetworkX_test(G, x)
        # delay = np.zeros((960, 960))
        # for i in range(960):
        #     for j in range(960):
        #         delay[i][j] = nx.dijkstra_path_length(G, i, j)
        # G_hop = nx.Graph()
        # G_hop = CreateNetworkX(G_hop, x)
        # # nx.draw(G_hop, with_labels=True)
        # # plt.show()
        # hop = np.zeros((960, 960))
        # for i in range(960):
        #     for j in range(960):
        #         hop[i][j] = nx.shortest_path_length(G_hop, i, j)
        # print(delay[0])
        # print(hop[0])
        _, _, _, c, t, l, r = F(len(amf), len(smf), len(upf), amf, smf, upf)
        capital.append(c)
        tptime.append(t)
        linkcost.append(l)
        reliability.append(r)
    print(capital)
    print(tptime)
    print(linkcost)
    print(reliability)
    x = [i for i in range(100)]
    fig = plt.figure()


    ax1 = fig.add_subplot(321)
    ax1.tick_params(direction='in')
    ax1.tick_params(top='on', right='on', which='both')

    ax1.set_title('capital')
    ax1.plot(x, capital, color="red", )
    ax1.plot(x, basic_line.basic_capital, color="blue")
    # plt.ylim((0, 107))

    ax2 = fig.add_subplot(322)
    ax2.set_title('tptime')
    ax2.plot(x, tptime, color="red", label='Sat CN', linewidth=1, marker="*")
    ax2.plot(x, basic_line.basic_tptime, color="blue", linestyle='--', label='Ground CN', linewidth=1, marker="+")
    ax2.set_xlabel('time')
    ax2.set_xlim((0, 107))
    ax2.set_ylabel('control plan procedure latency(ms)')
    plt.legend()

    ax3 = fig.add_subplot(323)
    ax3.set_title('linkcost')
    ax3.plot(x, linkcost, color="red", label='Sat CN', linewidth=1, marker="*")
    ax3.plot(x, basic_line.basic_linkcost, color="blue", linestyle='--', label='Ground CN', linewidth=1, marker="+")
    ax3.set_xlabel('time')
    ax3.set_xlim((0, 107))
    ax3.set_ylabel('linkCost')
    plt.legend()

    ax4 = fig.add_subplot(324)
    ax4.set_title('reliability')
    ax4.plot(x, reliability, color="red", label='Sat CN', linewidth=1, marker="*")
    ax4.plot(x, basic_line.basic_reliability, color="blue", linestyle='--', label='Ground CN', linewidth=1, marker="+")
    ax4.set_xlabel('time')
    ax4.set_xlim((0, 107))
    ax4.set_ylabel('control plan procedure robustness')
    plt.legend()

    ax5 = fig.add_subplot(325)
    weight = []
    weight_ground = []
    for i in range(100):
        weight.append(_mu * capital[i] + _nu * tptime[i] + _lambda * linkcost[i] + zeta * reliability[i])
        weight_ground.append(
            _mu * basic_line.basic_capital[i] + _nu * basic_line.basic_tptime[i] + _lambda * basic_line.basic_linkcost[
                i] + zeta * basic_line.basic_reliability[i])
    ax5.set_title('weighting')
    ax5.plot(x, weight, color="red", )
    ax5.plot(x, weight_ground, color="blue")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    if compute_nsga2:
        MAX_AMF_NUMBER = 15
        MAX_SMF_NUMBER = 15
        MAX_UPF_NUMBER = 15
        start_time = time.time()

        nsga2.suanfa(MAX_AMF_NUMBER, MAX_SMF_NUMBER, MAX_UPF_NUMBER)
        stop_time = time.time()
        print('time = ', stop_time - start_time)
