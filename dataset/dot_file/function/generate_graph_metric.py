import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np

def dot_to_nodes_and_edges(dot_file):
    graph = {"nodes":[],"edges":[]}
    with open(dot_file) as f:
        nodes = []
        edges = []
        for line in f:
            if ('label=' in line and 'name=' in line):
                temp = line.split(' ')
                node = {
                    "tag": temp[0].strip('}'),
                    "label":temp[1].split('=')[1].strip('"').strip(':'),
                    "def_name":temp[2].strip('"'),
                    "full_name":temp[3].split('=')[1].strip('"'),
                }
                print(node)
                nodes.append(node)
            if('->' in line):
                temp = line.split(' ')
                edges.append((temp[0],temp[2]))
                print((temp[0],temp[2]))
        graph["nodes"] = nodes
        graph["edges"] = edges
    return graph


def create_diGraph(graphInfo):
    nodesInfo = graphInfo["nodes"]
    edges = graphInfo["edges"]
    nodes = []
    for item in nodesInfo:
        nodes.append(item["tag"])
    G = nx.DiGraph()   
    G.add_nodes_from(nodes)
    for index in range(len(edges)):
        temp = (edges[index][0],edges[index][1],1)
        G.add_weighted_edges_from([temp])
    return G

def generate_metric(graph) -> dict["degree":any,"neighbors_count":any,"in_degree":any,"out_degree":any,"self_loops":any,"max_call_circles":any]:
    #以下的变量存储所有节点的图结构信息
    degree = [] #度数
    neighbors = [] #邻居
    neighbors_count = [] #邻居数
    in_degree = [] #入度数
    out_degree = [] #出度数
    is_self_loops = [] #是否自环
    max_call_circles = [] #最大调用环
    for item in nx.degree(graph):
        degree.append(item)
    for item in graph.in_degree:
        in_degree.append(item)
    for item in graph.out_degree:
        out_degree.append(item)
    self_loops = []
    for item in nx.nodes_with_selfloops(graph):
        self_loops.append(item)
    for node in graph.nodes:
        if(node in self_loops):
            is_self_loops.append((node,1))
        else:
            is_self_loops.append((node,0))
    for node in graph.nodes:
        neighbors_for_node = []
        for neighbor in graph.neighbors(node):
            neighbors_for_node.append(neighbor)
        neighbors.append({node:neighbors_for_node})
        neighbors_count.append((node,len(neighbors_for_node)))
    circles = []
    for circle in nx.simple_cycles(graph):
        circles.append(circle)
    for node in graph.nodes:
        max_call_circle = 0
        for circle in circles:
            if(node in circle):
                max_call_circle = max(max_call_circle,len(circle))
        max_call_circles.append((node,max_call_circle))

    return {"degree":degree,"neighbors_count":neighbors_count,"in_degree":in_degree,"out_degree":out_degree,"self_loops":is_self_loops,"max_call_circles":max_call_circles}

def generate_graph_metric():
    directory='./'
    target = '../../graph_metric/function/'
    subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]
    for dirName in subdirectories:
        arr = []
        for fileName in os.listdir(dirName):
            graphInfo = dot_to_nodes_and_edges(f"./{dirName}/{fileName}")
            graph = create_diGraph(graphInfo)
            metrics = generate_metric(graph)
            columns = ["full_name","label"] 
            for column in metrics.keys():
                columns.append(column)
            for index in range(len(graphInfo["nodes"])):
                full_name = '.'.join(graphInfo["nodes"][index]["full_name"].split('::'))
                row = [full_name,graphInfo["nodes"][index]["label"]]
                for metric in metrics.keys():
                    row.append(metrics[metric][index][1])
                arr.append(row)
        df = pd.DataFrame(arr,columns=columns)
        df.to_csv(target+f'{dirName}.csv',index=False)
        
        
# graphInfo = dot_to_nodes_and_edges(f"./scipy/scipy.dot")
# generate_metric(create_diGraph(graphInfo))
generate_graph_metric()
