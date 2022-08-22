import numpy as np
import matplotlib.pyplot as plt

k = 15 # 簇的数量
agent = 20 # 智能体的个数

dataset_path = r"pooled_mat.txt"
network_path = r"switch_network1.npz"
result_path = r"result.txt"

As = np.load(network_path)
As = [As['A1'], As['A2'], As['A3']]

Y = []
with open(dataset_path, 'r') as file:
    lines = file.readlines()
    for oneline in lines:
        oneline = oneline.split()
        count = 0
        for y in oneline:
            oneline[count] = float(y)
            count += 1
        oneline = np.array(oneline)
        Y.append(oneline)
Y = np.array(Y)

m = Y.shape[0]  # 数据的总个数
mi = int(m/agent) # 每个智能体分到的数据量

Yis = []
idx = np.arange(m)
np.random.shuffle(idx) # 打乱
for i in range(agent):
    Yis.append(Y[idx[i*mi:(i+1)*mi]]) # 每个智能体对应处理多少数据
Yis = np.array(Yis)

cik = []
first_num = np.random.randint(0,agent)
first_idx_ = np.random.randint(0,mi)
cik.append(Yis[first_num][first_idx_])  # 随机选取第一个聚类中心

for i in range(k-1):
    large_dis = []                    # 每个智能体处理的数据中与聚类中心距离最大值
    large_idx = []                    # 每个智能体里数据与聚类中心距离最大的索引
    for j in range(agent):            # j是智能体
        min_dc_dis = []
        for y in Yis[j]:              # y是每个智能体处理的数据
            data_cluster_dis = []     # 每个数据与所有聚类中心的距离
            for cluster in cik:       # cluster是每个聚类中心
                data_cluster_dis.append(np.linalg.norm(y-cluster))
            min_dc_dis.append(np.min(data_cluster_dis))
        min_dc_dis = np.array(min_dc_dis)
        large_dis.append(np.max(min_dc_dis))
        large_idx.append(np.argmax(min_dc_dis))
    large_dis = np.array(large_dis)
    large_agent_idx = np.argmax(large_dis)
    cik.append(Yis[large_agent_idx][large_idx[large_agent_idx]])
cik = np.array(cik)

m_th = 0.8
u_th = 0.2
s = 0

# 随时间变化
c = []
c.append(cik)
c.append(cik)
for t in range(10):
    mik = [[0 for i in range(k)] for j in range(agent)]  # 20*100 点的个数
    uik = [[np.zeros(Yis.shape[2]) for i in range(k)] for j in range(agent)]  # 数据的和
    Y_result = [[] for i in range(k)]
    for i in range(agent):
        last_cik = c[-1]  # 最新簇的位置
        for j in range(mi):
            distance = np.linalg.norm(last_cik - Yis[i][j], axis=1)
            cluster = np.argmin(distance)
            Y_result[cluster].append(Y[i][j])
            uik[i][cluster] += Yis[i][j]
            mik[i][cluster] += 1
    uik = np.array(uik)
    data_uik = []
    for i in range(uik.shape[2]):
        data_i = uik[:, :, i]
        data_uik.append(data_i)

    data_uik = np.array(data_uik)
    data_num = np.mat(mik)

    while True:
        try:
            A_temp = np.mat(As[s % 3])  # 获取通信拓扑结构 3n
            data_num = A_temp * data_num
            m_mean = data_num.mean(axis=0)
            m_dis = np.abs(data_num - m_mean).mean()  # m的总差距

            time = 0
            for data_i in data_uik:
                data_i = A_temp * data_i
                data_uik[time] = data_i
                time += 1
                ui_mean = data_i.mean(axis=0)
                ui_dis = np.abs(data_i - ui_mean).mean()
                #print(ui_dis)
                if ui_dis >= u_th:
                    break

            if m_dis <= m_th and time == data_uik.shape[0]:
                data_num[data_num == 0] = 1
                ci = []
                for data_i in data_uik:
                    cik_i = (data_i / data_num).A
                    ci.append(cik_i[0])
                ci = np.array(ci)
                ci = ci.reshape(k, uik.shape[2])
                c.append(ci)

                break
        finally:
            s += 1

print(c[1:])
'''
cik_result = c[2]
with open(result_path, 'w') as file:
    for i in range(len(cik_result)):
        file.write(str(cik_result[i])+'\n')
'''