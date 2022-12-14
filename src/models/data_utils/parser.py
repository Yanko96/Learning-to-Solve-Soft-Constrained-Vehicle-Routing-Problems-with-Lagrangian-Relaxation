import argparse
import os
import sys

import numpy as np
import pyparsing as pyp

from .Seq import *
from .utils import *


class vrpParser(object):
    def parse(self, problem, debug=False):
        self.is_debug = debug
        dm = VrpManager(problem['capacity'])
        dm.nodes.append(VrpNode(x=problem['depot'][0], y=problem['depot'][1], demand=0, px=problem['depot'][0], py=problem['depot'][1], capacity=problem['capacity'], dis=0.0))
        for customer in problem['customers']:
            dm.nodes.append(VrpNode(x=customer['position'][0], y=customer['position'][1], demand=customer['demand'], px=customer['position'][0], py=customer['position'][1], capacity=problem['capacity'], dis=0.0))
        dm.num_nodes = len(dm.nodes)
        cur_capacity = problem['capacity']
        pending_nodes = [i for i in range(0, dm.num_nodes)]
        dm.add_route_node(0)
        cur_capacity = dm.vehicle_state[-1][1]
        while len(pending_nodes) > 1:
            dis = []
            demands = []
            pre_node_idx = dm.vehicle_state[-1][0]
            pre_node = dm.get_node(pre_node_idx)
            for i in pending_nodes:
                cur_node = dm.get_node(i)
                dis.append(dm.get_dis(pre_node, cur_node))
                demands.append(cur_node.demand)
            for i in range(len(pending_nodes)):
                for j in range(i + 1, len(pending_nodes)):
                    if dis[i] > dis[j] or dis[i] == dis[j] and demands[i] > demands[j]:
                        pending_nodes[i], pending_nodes[j] = pending_nodes[j], pending_nodes[i]
                        dis[i], dis[j] = dis[j], dis[i]
                        demands[i], demands[j] = demands[j], demands[i]
            for i in pending_nodes:
                if i == 0:
                    if cur_capacity == problem['capacity']:
                        continue
                    dm.add_route_node(0)
                    break
                else:
                    cur_node = dm.get_node(i)
                    if cur_node.demand > cur_capacity:
                        continue
                    dm.add_route_node(i)
                    pending_nodes.remove(i)
                    break
            cur_capacity = dm.vehicle_state[-1][1]
        dm.add_route_node(0)
        return dm

class vrptwParser(object):
    def parse(self, problem, debug=False):
        self.is_debug = debug
        dm = VrptwManager(problem['capacity'])
        dm.nodes.append(VrptwNode(x=problem['depot'][0], y=problem['depot'][1], demand=0, tw_start=0, tw_end=9999, px=problem['depot'][0], py=problem['depot'][1], capacity=problem['capacity'], dis=0.0))
        for customer in problem['customers']:
            dm.nodes.append(VrptwNode(x=customer['position'][0], y=customer['position'][1], demand=customer['demand'], tw_start=customer['time_window'][0], tw_end=customer['time_window'][1], px=customer['position'][0], py=customer['position'][1], capacity=problem['capacity'], dis=0.0))
        dm.num_nodes = len(dm.nodes)
        cur_capacity = problem['capacity']
        pending_nodes = [i for i in range(0, dm.num_nodes)]
        dm.add_route_node(0)
        cur_capacity = dm.vehicle_state[-1][1]
        while len(pending_nodes) > 1:
            dis = []
            demands = []
            pre_node_idx = dm.vehicle_state[-1][0]
            pre_node = dm.get_node(pre_node_idx)
            for i in pending_nodes:
                cur_node = dm.get_node(i)
                dis.append(dm.get_dis(pre_node, cur_node))
                demands.append(cur_node.demand)
            for i in range(len(pending_nodes)):
                for j in range(i + 1, len(pending_nodes)):
                    if dis[i] > dis[j] or dis[i] == dis[j] and demands[i] > demands[j]:
                        pending_nodes[i], pending_nodes[j] = pending_nodes[j], pending_nodes[i]
                        dis[i], dis[j] = dis[j], dis[i]
                        demands[i], demands[j] = demands[j], demands[i]
            for i in pending_nodes:
                if i == 0:
                    if cur_capacity == problem['capacity']:
                        continue
                    dm.add_route_node(0)
                    break
                else:
                    cur_node = dm.get_node(i)
                    if cur_node.demand > cur_capacity:
                        continue
                    dm.add_route_node(i)
                    pending_nodes.remove(i)
                    break
            cur_capacity = dm.vehicle_state[-1][1]
        dm.add_route_node(0)
        dm.update_routes()
        return dm
