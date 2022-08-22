import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import math
import copy


def distance(a, b):
    return (math.fabs(a ** 2 - b ** 2)) ** 0.5


def Dis(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def top_1(A):  # 拓扑结构，邻接矩阵 one-hop
    G_top = [[0 for i in range(agent_n)] for j in range(len(A))]
    for i in range(agent_n):
        for j in range(i + 1, agent_n):
            if Dis(A[i], A[j]) <= rc:
                G_top[i][j] = 1
                G_top[j][i] = 1
    return np.mat(G_top)


def top_2(A, top1):  # two-hop
    G_top = [[0 for i in range(agent_n)] for j in range(agent_n)]
    for i in range(agent_n):
        for n in range(i + 1, agent_n):
            if top1[i, n] == 1:
                for m in range(agent_n):
                    if Dis(A[n], A[m]) <= rc:
                        G_top[i][m] = 1
                        G_top[m][i] = 1
    return np.mat(G_top) - top1


def B(top1, A):
    B_1 = [0 for m in range(agent_n)]
    s_rc = np.pi * rc ** 2
    r_j = rc / 2
    for i in range(agent_n):
        s = 0
        for j in range(agent_n):
            if top1[i, j] == 1:
                ij = Dis(A[i], A[j])
                if ij <= r_j:
                    s_j = np.pi * r_j ** 2
                if r_j < ij <= rc:
                    theta = math.acos((r_j ** 2 + ij ** 2 - rc ** 2) / (2 * r_j * ij))
                    theta_i = 2 * math.acos((ij ** 2 + rc ** 2 - r_j ** 2) / (2 * rc * ij))
                    s_theta_i = theta_i / (2 * np.pi) * rc ** 2 * np.pi
                    s_theta = (2 * np.pi - theta) / (2 * np.pi) * r_j ** 2 * np.pi
                    s_tri = rc / 2 * ij * np.sin(theta)
                    s_j = s_theta_i - s_tri + s_theta
                s += s_j
        B_1[i] = s / s_rc
    return B_1


def priority(B):
    P = [0 for m in range(agent_n)]
    ordered_B_1 = sorted(range(len(B)), key=lambda k: B[k])  # 得到排序后的索引
    B_2 = np.linspace(0, 1, agent_n)
    for i in range(agent_n):
        P[i] = B_2[ordered_B_1[i]]
    return np.array(P)


def isClusterhead(top1, p):
    ch_i = [0 for i in range(agent_n)]
    for i in range(agent_n):
        ch_i[i] = i
        for j in range(agent_n):
            if top1[i, j] == 1:
                if p[j] >= p[ch_i[i]]:
                    ch_i[i] = j
        for j in range(agent_n):
            if top1[i, j] == 1:
                if ch_i[j] == i:
                    G_type[i] = 1
    ch_workfor = np.zeros(np.shape(top1))
    for i in range(agent_n):
        if G_type[i] == 1:
            ch_workfor[i, i] = 1
            for j in range(i + 1, agent_n):
                if top1[i, j] == 1 and G_type[j] == 1:
                    ch_workfor[i, j] = 1
                    ch_workfor[j, i] = 1
    return ch_workfor


def isDoorway(top1, P):
    door_workfor = np.zeros(np.shape(top1))
    for i in range(agent_n):
        if G_type[i] == 1:
            continue
        for c1 in range(agent_n):
            if top1[i, c1] == 1 and G_type[c1] == 1:
                for p in range(agent_n):
                    if top1[i, p] == 1 and p != c1 and G_type[p] != 1:
                        for c2 in range(agent_n):
                            flag = 1

                            if top1[p, c2] == 1 and G_type[c2] == 1 and top1[i, c2] != 1 and top1[c1, c2] != 1:

                                for m in range(agent_n):
                                    if top1[i, m] == 1 and m != c1 and m != c2 and m != p:
                                        # case(5a)
                                        if top1[m, c1] == 1 and top1[m, c2] == 1:
                                            flag = 0  # 0则返回

                                        # case(5b,5c)
                                        if top1[m, c2] == 1:
                                            if G_type[m] == 1:
                                                flag = 0
                                            else:
                                                for n in range(agent_n):
                                                    if top1[c1, n] == 1 and top1[
                                                        m, n] == 1 and n != c1 and n != i and n != p and n != m and n != c2 and \
                                                            G_type[n] == 1:
                                                        flag = 0
                                                        break
                                        # case(5d)
                                        if top1[c2, m] == 1 and m != i:
                                            if P[m] > P[i]:
                                                flag = 0
                                            else:
                                                for n in range(agent_n):
                                                    if top1[c1, n] == 1 and top1[
                                                        m, n] == 1 and n != c1 and n != i and n != p and n != m and n != c2 and \
                                                            P[n] > P[i]:
                                                        flag = 0
                                                        break
                                        if flag == 0:
                                            break

                                if flag == 0:
                                    continue
                                G_type[i] = 2
                                door_workfor[i, i] = 1
                                door_workfor[i, c1] = 1
                                door_workfor[i, c2] = 1
    return door_workfor


def isGateway(top1, p):
    gate_workfor = np.zeros(np.shape(top1))
    for i in range(agent_n):
        if G_type[i] == 1 or G_type[i] == 2:
            continue
        for c1 in range(agent_n):
            if top1[i, c1] == 1 and (G_type[c1] == 1 or G_type[c1] == 2):
                for c2 in range(agent_n):
                    flag = 1
                    if top1[i, c2] == 1 and c2 != c1 and top1[c1, c2] != 1 and (
                            G_type[c2] == 1 or G_type[c2] == 2) and (G_type[c1] != 2 or G_type[c2] != 2):
                        for n in range(agent_n):
                            if top1[c1, n] == 1 and top1[c2, n] == 1 and n != i and n != c1 and n != c2:

                                # case 6a
                                if G_type[n] == 1 or G_type[n] == 2:
                                    flag = 0
                                # case 6b
                                if p[n] > p[i]:
                                    flag = 0

                                if flag == 0:
                                    break
                        if flag == 0:
                            continue
                        G_type[i] = 3
                        gate_workfor[i, i] = 1
                        gate_workfor[i, c1] = 1
                        gate_workfor[i, c2] = 1
    return gate_workfor


def pruning(top1, p, gate_workfor, door_workfor, ch_workfor):
    new_top = copy.deepcopy(top1)
    for i in range(agent_n):

        # case 7a
        if G_type[i] == 1:
            for m in range(agent_n):
                if top1[i, m] == 1 and G_type[m] == 1:
                    for n in range(agent_n):
                        if top1[i, n] == 1 and n != m and G_type[n] == 1:
                            if top1[n, m] == 1 and top1[m, n] == 1:
                                s = np.max([p[i], p[m], p[n]])
                                if s == p[i]:
                                    new_top[m, n] = 0
                                    new_top[n, m] = 0
                                    ch_workfor[n, m] = 0
                                    ch_workfor[m, n] = 0
        # case 7b
        if G_type[i] == 2:
            for m in range(agent_n):
                if top1[i, m] == 1 and G_type[m] == 1:
                    for n in range(agent_n):
                        if top1[i, n] == 1 and n != m and G_type[n] == 1:
                            if top1[n, m] == 1 and top1[m, n] == 1:
                                temp = m
                                if p[m] > p[n]:
                                    temp = n
                                new_top[temp, i] = 0
                                new_top[i, temp] = 0
                                door_workfor[i, temp] = 0
        # case 7c
        if G_type[i] == 3:
            nodes = []
            for m in range(agent_n):
                if top1[i, m] == 1 and G_type[m] == 1:
                    nodes.append(m)
            groups = []
            node = copy.deepcopy(nodes)
            for one_node in node:
                group = [one_node]
                for temp in group:
                    node_temp = []
                    for n in node:
                        if ch_workfor[temp, n] == 1 and n != temp:
                            group.append(n)
                            node_temp.append(n)
                    for n in node_temp:
                        node.remove(n)
                group = np.array(group)
                groups.append(group)
                if len(group) >= 2:
                    temp = group[0]
                    for k in range(1, len(group)):
                        if p[temp] < p[k]:
                            gate_workfor[i, temp] = 0
                            new_top[i, temp] = 0
                            new_top[temp, i] = 0
                            temp = k
    return new_top


def CWP(new_top, G):
    cwp = np.zeros(np.shape(G))
    for i in range(len(G)):
        temp = []
        temp_dis = []
        for s in range(len(G)):
            if new_top[i,s] == 1 or s == i:
                temp.append(G[s])
        for q in temp:
            temp_dis.append(Dis(q, np.array((0.,0.))))
        s = np.argmax(temp_dis)
        t = np.argmin(temp_dis)
        cwp[i] = (temp[s] + temp[t]) / 2
    return cwp


def update_G(G, top1, cwp):
    G_new = np.zeros(np.shape(G))
    for i in range(len(G)):
        for lamda in np.arange(1, 0, -rc / agent_n):
            flag = 1
            new_q = (1 - lamda) * G[i] + lamda * cwp[i]
            if Dis(new_q, G[i]) <= rc:
                for j in range(agent_n):
                    if top1[i,j] == 1:
                        center = (G[i] + G[j]) / 2
                        if Dis(new_q, center) > rc/2:
                            flag = 0
                            break
            if flag == 0:
                continue
            else:
                G_new[i] = new_q
                break
    return G_new


def HSBMAS(A):
    G_all = []
    top1_G = top_1(A)
    for t in range(60):
        #top2_G = top_2(A, top1_G)
        P = priority(B(top1_G, A))
        ch_workfor = isClusterhead(top1_G, P)
        door_workfor = isDoorway(top1_G, P)
        gate_workfor = isGateway(top1_G, P)
        top1_G = pruning(top1_G, P, gate_workfor, door_workfor, ch_workfor)
        cwp = CWP(top1_G, A)
        A = update_G(A, top1_G, cwp)
        G_all.append(A)
    return np.array(G_all)


def draw_3d(G_all):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    color = ['b', 'y', 'r', 'g', 'pink', 'orange', 'purple']
    z = np.arange(G_all.shape[0])
    for i in range(G_all.shape[1]):
        one_x = []
        one_y = []
        for j in range(G_all.shape[0]):
            one_x.append(G_all[j][i][0])
            one_y.append(G_all[j][i][1])
        one_x = np.array(one_x)
        one_y = np.array(one_y)
        ax.plot(one_x, one_y, z, c=color[i % 7])
    plt.show()


rc = 0.5
L = 10

G = np.load(r"D:\Documents\QG工作室\暑期学习\最终考核\数据集\Agents-200\001log_uniform_200\001log_uniform_200.npy")
agent_n = len(G)
G_type = np.array([0 for m in range(len(G))])  # 1-簇头，2-门路，3-网关，0-无骨干

G_all = HSBMAS(G)
draw_3d(G_all)
