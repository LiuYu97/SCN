import math
import os
import random
from satellite_settings import *
from start_stk import Start_STK
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import  distribute

import time
startTime = time.time()
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
from comtypes.gen import STKObjects, STKUtil, AgStkGatorLib
from comtypes.client import CreateObject, GetActiveObject, GetEvents, CoGetObject, ShowEvents
from ctypes import *
import comtypes.gen._00020430_0000_0000_C000_000000000046_0_2_0
from comtypes import GUID
from comtypes import helpstring
from comtypes import COMMETHOD
from comtypes import dispid
from ctypes.wintypes import VARIANT_BOOL
from ctypes import HRESULT
from comtypes import BSTR
from comtypes.automation import VARIANT
from comtypes.automation import _midlSAFEARRAY
from comtypes import CoClass
from comtypes import IUnknown
import comtypes.gen._00DD7BD4_53D5_4870_996B_8ADB8AF904FA_0_1_0
import comtypes.gen._8B49F426_4BF0_49F7_A59B_93961D83CB5D_0_1_0
from comtypes.automation import IDispatch
import comtypes.gen._42D2781B_8A06_4DB2_9969_72D6ABF01A72_0_1_0
from comtypes import DISPMETHOD, DISPPROPERTY, helpstring


"""
SET TO TRUE TO USE ENGINE, FALSE TO USE GUI
"""
useStkEngine = True
Read_Scenario = False
stkRoot = Start_STK(useStkEngine, Read_Scenario)
stkRoot.UnitPreferences.SetCurrentUnit("DateFormat", "EpSec")

print("Creating scenario...")
if not Read_Scenario:
    stkRoot.NewScenario('StarLink')
scenario = stkRoot.CurrentScenario
scenario2 = scenario.QueryInterface(STKObjects.IAgScenario)
# scenario2.StartTime = '24 Sep 2020 16:00:00.00'
# scenario2.StopTime = '25 Sep 2020 16:00:00.00'

totalTime = time.time() - startTime
splitTime = time.time()
print("--- Scenario creation: {a:4.3f} sec\t\tTotal time: {b:4.3f} sec ---".format(a=totalTime, b=totalTime))
Time_Range = 180000  # Seconds
Time_Step = 145.4  # Seconds
orbitNum = 24
satsNum = 40

AMF_NUMBER = 5
SMF_NUMBER = 5
UPF_NUMBER = 5
    
EbN0_Min_forward_and_backward = 0
EbN0_Min_left = 0
EbN0_Min_right = 0
G_sat = nx.Graph()
core_network_matrix = [['beijing', 39.92, 116.46], ['shanghai', 31.22, 121.48], ['wulumuqi', 43.47, 87.41], ['lasa', 29.6, 91], ['hainan', 20.02, 110.35]]

target_load = np.zeros([50, 50])
f = open('population.txt')
lines = f.readlines()
target_load_row = 0

for line in lines:
    list = line.strip('\n').split(' ')
    target_load[target_load_row, :] = list[0:50]
    target_load_row += 1


def Target_Position_And_Load_Matrix():

    Target_Position_Matrix = np.zeros((50, 50, 4))
    lat = 82.26
    lon = -176.4
    for i in range(Target_Position_Matrix.shape[0]):
        for j in range(Target_Position_Matrix.shape[1]):
            Target_Position_Matrix[i][j][0] = lat - i * 3.48
            Target_Position_Matrix[i][j][1] = lon + j * 7.2
            Target_Position_Matrix[i][j][3] = target_load[i][j]
    return Target_Position_Matrix


def Compute_Satellite_Position(current_time, sat):
    satLLADP = sat.DataProviders.GetDataPrvTimeVarFromPath('LLA State/Fixed')
    results_LLA = satLLADP.ExecSingleElements(current_time, ElementNames=["Lat", "Lon", "Alt"])
    Lat = results_LLA.DataSets.GetDataSetByName('Lat').GetValues()
    Lon = results_LLA.DataSets.GetDataSetByName('Lon').GetValues()
    Alt = results_LLA.DataSets.GetDataSetByName('Alt').GetValues()
    return Lat, Lon, Alt

def Get_Closest_Satellite(target, current_time):
    # x = h * cos(la) * cos(lo)
    # y = h * cos(la) * sin(lo)
    # z = h * sin(la)
    # h = 海拔高度 + 地球半径
    # la = 纬度 （弧度）
    # lo = 经度 （弧度）
    if target.InstanceName[:7] == "Gateway":
        Target_Position_Matrix = Target_Position_And_Load_Matrix()
        distance = []
        now_target_name = target.InstanceName
        tar_row = int(now_target_name.split('_')[1])
        tar_col = int(now_target_name.split('_')[2])

        for sat in tqdm(sat_list):
            sat_name = sat.InstanceName
            ear_h = 6371.004

            s_lat, s_lon, s_alt = Compute_Satellite_Position(current_time, sat)

            sat_h = ear_h + s_alt[0]
            sat_x = sat_h * math.cos(s_lat[0]) * math.cos(s_lon[0])
            sat_y = sat_h * math.cos(s_lat[0]) * math.sin(s_lon[0])
            sat_z = sat_h * math.sin(s_lat[0])

            t_lat = Target_Position_Matrix[tar_row][tar_col][0]
            t_lon = Target_Position_Matrix[tar_row][tar_col][1]
            tar_h = ear_h
            tar_x = tar_h * math.cos(t_lat) * math.cos(t_lon)
            tar_y = tar_h * math.cos(t_lat) * math.sin(t_lon)
            tar_z = tar_h * math.sin(t_lat)
            distance.append(
                (sat_name, math.sqrt(pow(sat_x - tar_x, 2) + pow(sat_y - tar_y, 2) + pow(sat_z - tar_z, 2))))
        distance2 = []
        for i in range(len(distance)):
            distance2.append(distance[i][1])
        return distance[distance2.index(min(distance2))]
    else:
        distance = []
        now_target_name = target.InstanceName
        core_num = int(now_target_name.split('_')[1])
        for sat in tqdm(sat_list):
            sat_name = sat.InstanceName
            ear_h = 6371.004

            s_lat, s_lon, s_alt = Compute_Satellite_Position(current_time, sat)

            sat_h = ear_h + s_alt[0]
            sat_x = sat_h * math.cos(s_lat[0]) * math.cos(s_lon[0])
            sat_y = sat_h * math.cos(s_lat[0]) * math.sin(s_lon[0])
            sat_z = sat_h * math.sin(s_lat[0])

            core_lat = core_network_matrix[core_num][1]
            core_lon = core_network_matrix[core_num][2]
            core_h = ear_h
            core_x = core_h * math.cos(core_lat) * math.cos(core_lon)
            core_y = core_h * math.cos(core_lat) * math.sin(core_lon)
            core_z = core_h * math.sin(core_lat)
            distance.append(
                (sat_name, math.sqrt(pow(sat_x - core_x, 2) + pow(sat_y - core_y, 2) + pow(sat_z - core_z, 2))))
        distance2 = []
        for i in range(len(distance)):
            distance2.append(distance[i][1])
        return distance[distance2.index(min(distance2))]

def Create_Ground():
    target_position_and_load = Target_Position_And_Load_Matrix()
    print('create gateway')
    for i in range(target_position_and_load.shape[0]):
        for j in range(target_position_and_load.shape[1]):
            target_name = 'Gateway' + '_' + str(i) + '_' + str(j)
            gateway = scenario.Children.New(STKObjects.eTarget, target_name)
            gateway2 = gateway.QueryInterface(STKObjects.IAgTarget)
            gateway2.Position.AssignGeodetic(target_position_and_load[i][j][0], target_position_and_load[i][j][1], 0)
    print('create gateway done')
    print('create corenetwork')
    for i in range(len(core_network_matrix)):
        target_name = 'CoreNetwork' + '_' + str(i)
        corenetwork = scenario.Children.New(STKObjects.eTarget, target_name)
        corenetwork2 = corenetwork.QueryInterface(STKObjects.IAgTarget)
        corenetwork2.Position.AssignGeodetic(core_network_matrix[i][1], core_network_matrix[i][2], 0)
    print('create corenetwork done')

    Ground_list = stkRoot.CurrentScenario.Children.GetElements(STKObjects.eTarget)

    ground_dic = {}
    for target in tqdm(Ground_list):
        ground_dic[target.InstanceName] = target
        # print(target.InstanceName)
    return ground_dic

def Create_Sat_Dic(sat_list, orbitNum, satsNum):
    # 创建卫星的字典，方便根据名字对卫星进行查找
    sat_dic = {}
    print('Creating Satellite Dictionary')
    for sat in tqdm(sat_list):
        sat_dic[sat.InstanceName] = sat
    Plane_num = []
    for i in range(0, orbitNum):
        Plane_num.append(i)
    Sat_num = []
    for i in range(0, satsNum):
        Sat_num.append(i)

    print("sat_dic len", len(sat_dic))
    print("Total satellite number:", len(sat_dic))
    print("plane_num", Plane_num)
    print("Sat_num", Sat_num)
    return sat_dic

def Export_Satellite_Matrix(n):
    Ground_list = stkRoot.CurrentScenario.Children.GetElements(STKObjects.eTarget)
    sat_list = stkRoot.CurrentScenario.Children.GetElements(STKObjects.eSatellite)
    current_time = scenario2.StartTime + n * Time_Step
    if current_time > scenario2.StartTime + Time_Range:
        return 0

    Adjacency_Matrix = np.zeros(
        [2 * len(sat_list) + AMF_NUMBER + SMF_NUMBER + UPF_NUMBER,
         2 * len(sat_list) + AMF_NUMBER + SMF_NUMBER + UPF_NUMBER])
    Delay_Matrix = np.zeros([len(sat_list), len(sat_list) ])
    BER_Matrix = np.zeros([len(sat_list) , len(sat_list) ])
    Rate_Matrix = np.zeros([len(sat_list), len(sat_list)])
    print("邻接矩阵shape： ", Adjacency_Matrix.shape, Delay_Matrix.shape, BER_Matrix.shape, Rate_Matrix.shape)

    for client in range(len(sat_list)):
        Adjacency_Matrix[len(sat_list) + client][client] = 1
        Adjacency_Matrix[client][len(sat_list) + client] = 1

    # 计算地面核心网与卫星的连接关系

    AMF_location_index = []
    SMF_location_index = []
    UPF_location_index = []



    for target in Ground_list:
        if target.InstanceName[:4] == "Core":
            now_target_name = target.InstanceName
            now_core_num = int(now_target_name.split('_')[1])
            closest_sat = Get_Closest_Satellite(target, current_time)
            closest_sat_name = closest_sat[0]
            now_plane_num = int(closest_sat_name.split('_')[0][3:])
            now_sat_num = int(closest_sat_name.split('_')[1])
            AMF_location_index.append(now_plane_num * satsNum + now_sat_num)
            SMF_location_index.append(now_plane_num * satsNum + now_sat_num)
            UPF_location_index.append(now_plane_num * satsNum + now_sat_num)
            Adjacency_Matrix[2 * len(sat_list) + now_core_num][now_plane_num * satsNum + now_sat_num] = 1
            Adjacency_Matrix[2 * len(sat_list) + now_core_num + 5][now_plane_num * satsNum + now_sat_num] = 1
            Adjacency_Matrix[2 * len(sat_list) + now_core_num + 10][now_plane_num * satsNum + now_sat_num] = 1

    # print("AMF_location", AMF_location_index)
    # print("SMF_location", SMF_location_index)
    # print("UPF_location", UPF_location_index)
    # np.savetxt("AMF_location.txt", AMF_location_index)
    np.savetxt("./data_ns3/" + str(n) + "AMF_location.txt", AMF_location_index, fmt="%d")

    for sat_num, sat in enumerate(sat_list):
        now_sat_name = sat.InstanceName
        now_sat_transmitter = sat.Children.GetElements(STKObjects.eTransmitter)[0]  # 找到该卫星的发射机
        Set_Transmitter_Parameter(now_sat_transmitter, frequency=12, EIRP=20, DataRate=14)
        # # 发射机与接收机相连
        # # 与后面的卫星的接收机相连
        # 计算前后的链路信息
        now_plane_num = int(now_sat_name.split('_')[0][3:])
        now_sat_num = int(now_sat_name.split('_')[1])
        access_backward = now_sat_transmitter.GetAccessToObject(
            Get_sat_receiver(sat_dic['Sat' + str(now_plane_num) + '_' + str((now_sat_num + 1) % satsNum)]))
        access_forward = now_sat_transmitter.GetAccessToObject(
            Get_sat_receiver(sat_dic['Sat' + str(now_plane_num) + '_' + str((now_sat_num - 1) % satsNum)]))

        Propagation_Delay_forward = Compute_Propagation_Delay(access_forward, current_time)
        Propagation_Delay_backward = Compute_Propagation_Delay(access_backward, current_time)

        BER_forward = Compute_BER(access_forward, current_time)
        BER_backward = Compute_BER(access_backward, current_time)

        if current_time == scenario2.StartTime:
            global EbN0_Min_forward_and_backward
            EbN0_Min_forward_and_backward = Compute_Min_EbN0(scenario2, access_forward, n=0)
        # print('EbN0_Min_forward_and_backward   ', EbN0_Min_forward_and_backward)

        EbN0_Min_forward_and_backward = Compute_Min_EbN0(scenario2, access_forward, n=0)
        Rate_forward = Compute_Rate(access_forward, current_time, EbN0_Min_forward_and_backward)
        Rate_backward = Compute_Rate(access_backward, current_time, EbN0_Min_forward_and_backward)


        if now_sat_num == 0:
            Adjacency_Matrix[sat_num][now_plane_num * satsNum + now_sat_num + 1] = 1
            Adjacency_Matrix[sat_num][now_plane_num * satsNum + satsNum - 1] = 1
            Delay_Matrix[sat_num][now_plane_num * satsNum + now_sat_num + 1] = Propagation_Delay_forward
            Delay_Matrix[sat_num][now_plane_num * satsNum + satsNum - 1] = Propagation_Delay_backward
            BER_Matrix[sat_num][now_plane_num * satsNum + now_sat_num + 1] = BER_forward
            BER_Matrix[sat_num][now_plane_num * satsNum + satsNum - 1] = BER_backward
            Rate_Matrix[sat_num][now_plane_num * satsNum + now_sat_num + 1] = Rate_forward
            Rate_Matrix[sat_num][now_plane_num * satsNum + satsNum - 1] = Rate_backward

            # Adjacency_Matrix[sat_num][now_sat_num+1] = Propagation_Delay_forward
            # Adjacency_Matrix[sat_num][10-1] = Propagation_Delay_backward
        elif now_sat_num == satsNum - 1:
            Adjacency_Matrix[sat_num][now_plane_num * satsNum + now_sat_num - 1] = 1
            Adjacency_Matrix[sat_num][now_plane_num * satsNum + 0] = 1
            Delay_Matrix[sat_num][now_plane_num * satsNum + now_sat_num - 1] = Propagation_Delay_backward
            Delay_Matrix[sat_num][now_plane_num * satsNum + 0] = Propagation_Delay_forward
            BER_Matrix[sat_num][now_plane_num * satsNum + now_sat_num - 1] = BER_forward
            BER_Matrix[sat_num][now_plane_num * satsNum + 0] = BER_backward
            Rate_Matrix[sat_num][now_plane_num * satsNum + now_sat_num - 1] = Rate_forward
            Rate_Matrix[sat_num][now_plane_num * satsNum + 0] = Rate_backward

            # Adjacency_Matrix[sat_num][0] = Propagation_Delay_forward
            # Adjacency_Matrix[sat_num][now_sat_num-1] = Propagation_Delay_backward
        else:
            Adjacency_Matrix[sat_num][now_plane_num * satsNum + now_sat_num - 1] = 1
            Adjacency_Matrix[sat_num][now_plane_num * satsNum + now_sat_num + 1] = 1
            Delay_Matrix[sat_num][now_plane_num * satsNum + now_sat_num - 1] = Propagation_Delay_backward
            Delay_Matrix[sat_num][now_plane_num * satsNum + now_sat_num + 1] = Propagation_Delay_forward
            BER_Matrix[sat_num][now_plane_num * satsNum + now_sat_num - 1] = BER_backward
            BER_Matrix[sat_num][now_plane_num * satsNum + now_sat_num + 1] = BER_forward
            Rate_Matrix[sat_num][now_plane_num * satsNum + now_sat_num - 1] = Rate_backward
            Rate_Matrix[sat_num][now_plane_num * satsNum + now_sat_num + 1] = Rate_forward

            # Adjacency_Matrix[sat_num][now_sat_num + 1] = Propagation_Delay_forward
            # Adjacency_Matrix[sat_num][now_sat_num - 1] = Propagation_Delay_backward

        # print(Adjacency_Matrix[sat_num])
        # 计算左右的链路信息，如果纬度大于75，无星间链路，如果是0好轨道，只有右边，因为左边是轨道缝
        # 默认两侧没有连接
        Propagation_Delay_left = 0
        Propagation_Delay_right = 0
        BER_left = 0
        BER_right = 0
        Rate_left = 0
        Rate_right = 0
        # 创建空列表存放卫星每个时刻
        X_List = []
        Y_List = []
        Z_List = []
        Time_List = []
        # 获得每个每个卫星的纬度
        Lat,_,_ = Compute_Satellite_Position(current_time, sat)
        # print(time[0], "   ", now_sat_name, "   纬度:  ", Lat[0], " X(", X[0], ")  Y(", Y[0], ")  Z(", Z[0], ")")
        if float(Lat[0]) > 75 or float(Lat[0]) < -75:
            # print("卫星纬度大于 75，无星间链路")
            pass
        else:
            # 计算左侧链路信息
            if now_plane_num != 0:
                access_left = now_sat_transmitter.GetAccessToObject(
                    Get_sat_receiver(sat_dic['Sat' + str((now_plane_num - 1) % orbitNum) + '_' + str(now_sat_num)]))
                Propagation_Delay_left = Compute_Propagation_Delay(access_left, current_time)
                BER_left = Compute_BER(access_left, current_time)

                if current_time == scenario2.StartTime:
                    global EbN0_Min_left
                    EbN0_Min_left = Compute_Min_EbN0(scenario2, access_left, n=0)
                # print('EbN0_Min_left   ', EbN0_Min_left)

                Rate_left = Compute_Rate(access_left, current_time, EbN0_Min_left)

                Adjacency_Matrix[sat_num][(now_plane_num - 1) * satsNum + now_sat_num] = 1
                Delay_Matrix[sat_num][(now_plane_num - 1) * satsNum + now_sat_num] = Propagation_Delay_left
                BER_Matrix[sat_num][(now_plane_num - 1) * satsNum + now_sat_num] = BER_left
                Rate_Matrix[sat_num][(now_plane_num - 1) * satsNum + now_sat_num] = Rate_left

            if now_plane_num != orbitNum - 1:
                # 计算右侧链路信息
                access_right = now_sat_transmitter.GetAccessToObject(
                    Get_sat_receiver(sat_dic['Sat' + str((now_plane_num + 1) % orbitNum) + '_' + str(now_sat_num)]))
                Propagation_Delay_right = Compute_Propagation_Delay(access_right, current_time)
                BER_right = Compute_BER(access_right, current_time)

                if current_time == scenario2.StartTime:
                    global EbN0_Min_right
                    EbN0_Min_right = Compute_Min_EbN0(scenario2, access_right, n=0)
                # print('EbN0_Min_right   ', EbN0_Min_right)

                Rate_right = Compute_Rate(access_right, current_time, EbN0_Min_right)

                Adjacency_Matrix[sat_num][(now_plane_num + 1) * satsNum + now_sat_num] = 1
                Delay_Matrix[sat_num][(now_plane_num + 1) * satsNum + now_sat_num] = Propagation_Delay_right
                BER_Matrix[sat_num][(now_plane_num + 1) * satsNum + now_sat_num] = BER_right
                Rate_Matrix[sat_num][(now_plane_num + 1) * satsNum + now_sat_num] = Rate_right

        # print("   ", now_sat_name, "Delay_forward", Propagation_Delay_forward, "  Delay_backward",
        #       Propagation_Delay_backward,
        #       "  Delay_left", Propagation_Delay_left, "  Delay_right", Propagation_Delay_right)
        # print("   ", now_sat_name, "BER_forward", BER_forward, "  BER_backward",
        #       BER_backward,
        #       "  BER_left", BER_left, "  BER_right", BER_right)
        # print("   ", now_sat_name, "Rate_forward", Rate_forward, "  Rate_backward",
        #       Rate_backward,
        #       "  Rate_left", Rate_left, "  Rate_right", Rate_right)



    # # 每个卫星最近的amf
    # Sat_shortest_AMF = np.zeros([len(sat_list), AMF_NUMBER])
    # AMF_shortest_SMF = np.zeros([AMF_NUMBER, SMF_NUMBER])
    # SMF_shortest_UPF = np.zeros([SMF_NUMBER, UPF_NUMBER])
    #
    # # 根据邻接矩阵生成networkx
    # G = CreateNetworkX(G_sat, Adjacency_Matrix)
    # for sat in range(len(sat_list)):
    #     for amf in range(len(AMF_location_index)):
    #         Sat_shortest_AMF[sat][amf] = nx.shortest_path_length(G, source=sat, target=AMF_location_index[amf])
    # for amf in range(len(AMF_location_index)):
    #     for smf in range(len(SMF_location_index)):
    #         AMF_shortest_SMF[amf][smf] = nx.shortest_path_length(G, source=AMF_location_index[amf], target=SMF_location_index[smf])
    # for smf in range(len(SMF_location_index)):
    #     for upf in range(len(UPF_location_index)):
    #         SMF_shortest_UPF[smf][upf] = nx.shortest_path_length(G, source=SMF_location_index[smf], target=UPF_location_index[upf])

    np.savetxt("./data_ns3/"+str(n) + "Adjacency_Matrix.txt", Adjacency_Matrix, fmt="%d")
    np.savetxt("./data_ns3/"+str(n) + "Delay_Matrix.txt", Delay_Matrix)
    np.savetxt("./data_ns3/"+str(n) + "BER_Matrix.txt", BER_Matrix)
    np.savetxt("./data_ns3/"+str(n) + "Rate_Matrix.txt", Rate_Matrix)
    # np.savetxt("./data_ns3/" + str(n) + "Sat_shortest_AMF.txt", Sat_shortest_AMF, fmt="%d")
    # np.savetxt("./data_ns3/" + str(n) + "AMF_shortest_SMF.txt", AMF_shortest_SMF, fmt="%d")
    # np.savetxt("./data_ns3/" + str(n) + "SMF_shortest_UPF.txt", SMF_shortest_UPF, fmt="%d")
    print("  is ok ")

def Get_Satellite_Load_Matrix(n):
    current_time = scenario2.StartTime + n * Time_Step
    if current_time > scenario2.StartTime + Time_Range:
        return 0

    Load_Matrix = np.zeros((orbitNum, satsNum))
    Ground_list = stkRoot.CurrentScenario.Children.GetElements(STKObjects.eTarget)
    for target in Ground_list:
        if target.InstanceName[:7] == "Gateway":
            closest_sat_name = Get_Closest_Satellite(target, current_time)
            # print(closest_sat_name, ' ', closest_sat_name[0], ' ', closest_sat_name[1])
            now_plane_num = int(closest_sat_name[0].split('_')[0][3:])
            now_sat_num = int(closest_sat_name[0].split('_')[1])
            # print(now_plane_num, " ", now_sat_num)
            now_target_name = target.InstanceName
            tar_row = int(now_target_name.split('_')[1])
            tar_col = int(now_target_name.split('_')[2])
            Load_Matrix[now_plane_num][now_sat_num] += target_load[tar_row][tar_col]
    return Load_Matrix

def CreateNetworkX(G_networkx, G_Matrix):
    for i in range(len(G_Matrix)):
        for j in range(len(G_Matrix)):
            if G_Matrix[i][j] == 1:
                G_networkx.add_edge(i, j)
    return G_networkx

if not Read_Scenario:
    Creat_satellite(scenario, numOrbitPlanes=orbitNum, numSatsPerPlane=satsNum, hight=620, Inclination=90)  # Starlink
    sat_list = stkRoot.CurrentScenario.Children.GetElements(STKObjects.eSatellite)
    Add_transmitter_receiver(sat_list)


if __name__ == '__main__':
    ground_dic = Create_Ground()
    sat_dic = Create_Sat_Dic(sat_list, orbitNum, satsNum)
    # 卫星负载矩阵
    # Sat_load = Get_Satellite_Load_Matrix(0)
    # sat_load = np.transpose(Sat_load)
    # np.savetxt("./data_ns3/Sat_load.txt", sat_load, fmt="%d")

    n = 595
    while(n <= 600):
        Sat_load = Get_Satellite_Load_Matrix(n)
        sat_load = np.transpose(Sat_load)
        np.savetxt("./data_ns3/" + str(n) + "Sat_load.txt", sat_load, fmt="%d")
        if Export_Satellite_Matrix(n) == 0:
            break
        n = n + 1
