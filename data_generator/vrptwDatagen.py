import argparse
import json
import os
import sys
from operator import itemgetter

import numpy as np
from scipy.spatial import distance_matrix
from tqdm import tqdm

argParser = argparse.ArgumentParser()
argParser.add_argument('--res_file', type=str, default='vrptw_20_30.json')
argParser.add_argument('--res_train_file', type=str, default='vrptw_20_30_train.json')
argParser.add_argument('--res_val_file', type=str, default='vrptw_20_30_val.json')
argParser.add_argument('--res_test_file', type=str, default='vrptw_20_30_test.json')
argParser.add_argument('--num_samples', type=int, default=100000)
argParser.add_argument('--seed', type=int, default=None)
argParser.add_argument('--num_customers', type=int, default=20)
argParser.add_argument('--max_demand', type=int, default=9)
argParser.add_argument('--position_range', type=float, default=1.0)
argParser.add_argument('--capacity', type=int, default=30, choices=[20, 30, 40, 50])
argParser.add_argument('--min_window_width', type=int, default=0.25)
argParser.add_argument('--mean_window_width', type=int, default=1)

args = argParser.parse_args()


def sample_pos():
    return np.random.rand(), np.random.rand()

class Vrp():

    # -----------初始数据定义---------------------

    def __init__(self, num_customers, capacity, distance, q):
        self.mans = num_customers                                       # 客户数量
        self.tons = capacity                                            # 车辆载重
        self.distance = distance                                        # 各个客户及配送中心距离
        self.q = q									                    # 8个客户分布需要的货物的需求量，第0位为配送中心自己
        self.savings = []                                               # 节约度
        self.Routes = []                                                # 路线
        self.Cost = 0                                                   # 总路程

    # -----------节约算法主程序---------------------

    def savingsAlgorithms(self):
        saving = 0
        for i in range(1, len(self.q)):
            self.Routes.append([i])

        for i in range(1, len(self.Routes) + 1):                                                 # 使用Sij = Ci0 + C0j - Cij计算节约度
            for j in range(1, len(self.Routes) + 1):
                if i == j:
                    pass
                else:
                    saving = (self.distance[i][0] + self.distance[0][j]) - self.distance[i][j]
                    self.savings.append([i, j, saving])                                          # 将结果以元组形式存放在列表中

        self.savings = sorted(self.savings, key=itemgetter(2), reverse=True)                     # 按照节约度从大到小进行排序
        # for i in range(len(self.savings)):
            # print(self.savings[i][0],'--',self.savings[i][1], "  ",self.savings[i][2])           # 打印节约度

        for i in range(len(self.savings)):
            startRoute = []
            endRoute = []
            routeDemand = 0
            for j in range(len(self.Routes)):
                if (self.savings[i][0] == self.Routes[j][-1]):
                    endRoute = self.Routes[j]
                elif (self.savings[i][1] == self.Routes[j][0]):
                    startRoute = self.Routes[j]

                if ((len(startRoute) != 0) and (len(endRoute) != 0)):
                    for k in range(len(startRoute)):
                        routeDemand += self.q[startRoute[k]]
                    for k in range(len(endRoute)):
                        routeDemand += self.q[endRoute[k]]
                    routeDistance = 0
                    routestore = [0]+endRoute+startRoute+[0]
                    for i in range(len(routestore)-1):
                        # print(routestore[i],routestore[i+1])
                        # print(self.distance[routestore[i]][routestore[i+1]])
                        routeDistance += self.distance[routestore[i]][routestore[i+1]]

                    #print(routestore,"== ==:",routeDistance)

                    if (routeDemand <= self.tons):     # 按照限制规则对路线进行更改
                        self.Routes.remove(startRoute)
                        self.Routes.remove(endRoute)
                        self.Routes.append(endRoute + startRoute)
                    break

        for i in range(len(self.Routes)):
            self.Routes[i].insert(0, 0)
            self.Routes[i].insert(len(self.Routes[i]), 0)

    # -----------输出最终结果---------------------

    def printRoutes(self):
        for i in self.Routes:
            costs = 0
            for j in range(len(i)-1):
                costs += self.distance[i[j]][i[j+1]]
            print("路线:  ",i,"  路程:  ",costs)

    def calcCosts(self):
        for i in range(len(self.Routes)):
            for j in range(len(self.Routes[i]) - 1):
                self.Cost += self.distance[self.Routes[i][j]][self.Routes[i][j + 1]]

        print("\nTotal Distance: ", round(self.Cost, 3))

    # -----------Master函数---------------------

    def start(self):                      # Master函数，调用所有其他函数
        # print("== == == 距离表 == == ==")
        # for i in self.distance:
        #     print(i)
        # print("== == == 需求表 == == ==")
        # print(self.q)
        # print("== == == 限制条件 == == ==")
        # print("车辆最大载重：",self.tons)
        # print("== == == == == == == == == == == == == == == 节约度 == == == == == == == = == == == == == == == =")
        self.savingsAlgorithms()          # 函数调用计算节省量并生成路线
        # print("== == == == == == == == == == == == == == == 结果 == == == == == == == = == == == == == == == =")
        # self.printRoutes()
        # self.calcCosts()
        return self.Routes


def generate_time_window(routes, dm):
    time_window = [[0.0, 0.0] for _ in range(dm.shape[0])]
    for route in routes:
        cur_dis = 0.0
        for i in range(len(route)-1):
            cur_dis += dm[route[i], route[i+1]]
            window_width = max(0.5, np.random.normal(loc=1, scale=1/3))
            time_window[route[i+1]] = [max(cur_dis-window_width, 0), cur_dis+window_width]
    return time_window


def main():
    np.random.seed(args.seed)
    samples = []
    for _ in tqdm(range(args.num_samples)):
        cur_sample = {}
        cur_sample['customers'] = []
        cur_sample['capacity'] = args.capacity
        dx, dy = sample_pos()
        cur_sample['depot'] = (dx, dy)
        for i in range(args.num_customers):
            cx, cy = sample_pos()
            demand = np.random.randint(1, args.max_demand + 1)
            cur_sample['customers'].append({'position': (cx, cy), 'demand': demand})
        
        node_positions = [[cur_sample['depot'][0], cur_sample['depot'][1]]] + [customer["position"] for customer in cur_sample['customers']]
        dm = distance_matrix(node_positions, node_positions)
        cw_solver = Vrp(20, 30, dm, [0] + [customer["demand"] for customer in cur_sample['customers']])
        routes = cw_solver.start()
        time_windows = generate_time_window(routes, dm)
        for i in range(len(time_windows)-1):
            cur_sample['customers'][i]["time_window"] = time_windows[i+1]

        samples.append(cur_sample)

    path = '../data/vrptw/'
    if not os.path.exists(path):
        os.makedirs(path)

    data_size = len(samples)
    print(data_size)
    fout_res = open(path+args.res_file, 'w')
    json.dump(samples, fout_res)

    fout_train = open(path+args.res_train_file, 'w')
    train_data_size = int(data_size * 0.8)
    json.dump(samples[:train_data_size], fout_train)

    fout_val = open(path+args.res_val_file, 'w')
    val_data_size = int(data_size * 0.9) - train_data_size
    json.dump(samples[train_data_size: train_data_size + val_data_size], fout_val)

    fout_test = open(path+args.res_test_file, 'w')
    test_data_size = data_size - train_data_size - val_data_size
    json.dump(samples[train_data_size + val_data_size:], fout_test)


main()
