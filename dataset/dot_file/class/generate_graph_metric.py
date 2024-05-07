import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
def dot_to_nodes_and_edges(dot_file):
    graph = {"nodes":[],"edges":[],"weights":[],"Instantiates":[]}
    with open(dot_file) as f:
        nodes = []
        edges = []
        weights = []
        Instantiates = []
        for line in f:
            if ('label=' in line and not '->' in line):
                temp = line.split(' ')
                node = {
                    "class_name":temp[0].strip('"'),
                }
                print(node)
                nodes.append(node)
            if('->' in line):
                temp = line.split(' ')
                class_relation = ''
                arrowType = temp[3].split('=')[1].strip(',').strip('"')
                if(arrowType=='empty'):
                    class_relation = 'inherit'
                elif(arrowType=='diamond'):
                    class_relation = 'combinate'
                elif(arrowType=='odiamond'):
                    class_relation = 'aggregate'
                edges.append((temp[0].strip('"'),temp[2].strip('"')))
                weights.append((temp[0].strip('"'),temp[2].strip('"'),class_relation))
                try:
                    Instantiates.append((temp[0].strip('"'),temp[2].strip('"'),temp[6].split('=')[1].strip(',').strip('"')))
                except:
                    pass
                print((temp[0].strip('"'),temp[2].strip('"')))
                
        graph["nodes"] = nodes
        graph["edges"] = edges
        graph["weights"] = weights
        graph["Instantiates"] = Instantiates
    return graph

def create_diGraph(graphInfo:dict["nodes":any,"edges":any,"weights":any,"Instantiates":any]):
    nodesInfo = graphInfo["nodes"]
    edges = graphInfo["edges"]
    weights = graphInfo["weights"]
    nodes = []
    for item in nodesInfo:
        nodes.append(item["class_name"])
    G = nx.DiGraph()   
    G.add_nodes_from(nodes)
    for index in range(len(edges)):
        G.add_weighted_edges_from([(edges[index][0],edges[index][1],1)],weight=weights[index][2])
    return G

def generate_metric(graph) -> dict["degree":any,"neighbors":any,"in_degree":any,"out_degree":any,"self_loops":any]:
    #以下的变量存储所有节点的图结构信息
    degree = [] #度数
    neighbors = [] #邻居数
    neighbors_count = []
    in_degree = [] #入度数
    out_degree = [] #出度数
    is_self_loops = [] #是否自环
    nodes_inherit = [] #继承的数量
    nodes_inherited = [] #被继承的数量
    nodes_aggregate = [] #聚合的数量
    nodes_aggregated = [] #被聚合的数量
    nodes_combinate = [] #组合的数量
    nodes_combinated = [] #被组合的数量
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
    for node in graph.nodes:
        node_inherit = 0
        node_inherited = 0
        node_aggregate = 0
        node_aggregated = 0
        node_combinate = 0
        node_combinated = 0
        for edge in graph.edges(data=True):
            if(edge[0]==node):
                try:
                    node_inherit += edge[2]['inherit']
                except:
                    pass
                try:
                    node_aggregate += edge[2]['aggregate']
                except:
                    pass
                try:
                    node_combinate += edge[2]['combinate']
                except:
                    pass
            if(edge[1]==node):
                try:
                    node_inherited += edge[2]['inherit']
                except:
                    pass
                try:
                    node_aggregated += edge[2]['aggregate']
                except:
                    pass
                try:
                    node_combinated += edge[2]['combinate']
                except:
                    pass
        nodes_inherit.append((node,node_inherit))
        nodes_inherited.append((node,node_inherited))
        nodes_aggregate.append((node,node_aggregate))
        nodes_aggregated.append((node,node_aggregated))
        nodes_combinate.append((node,node_combinate))
        nodes_combinated.append((node,node_combinated))
    return {"degree":degree,"neighbors_count":neighbors_count,"in_degree":in_degree,"out_degree":out_degree,"self_loops":is_self_loops,"nodes_inherit": nodes_inherit,"nodes_inherited":nodes_inherited,"nodes_aggregate":nodes_aggregate, "nodes_aggregated":nodes_aggregated,"nodes_combinate":nodes_combinate,"nodes_combinated":nodes_combinated}

def generate_graph_metric():
    directory='./'
    target = '../../graph_metric/class/'
    subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]
    for dirName in subdirectories:
        arr = []
        for fileName in os.listdir(dirName):
            graphInfo = dot_to_nodes_and_edges(f"./{dirName}/{fileName}")
            graph = create_diGraph(graphInfo)
            metrics = generate_metric(graph)
            columns = ["class_name"]
            for column in metrics.keys():
                columns.append(column)
            for index in range(len(graphInfo["nodes"])):
                row = [graphInfo["nodes"][index]["class_name"]]
                for metric in metrics.keys():
                    print(metric)
                    row.append(metrics[metric][index][1])
                arr.append(row)
        df = pd.DataFrame(arr,columns=columns)
        df.to_csv(target+f'{dirName}.csv',index=False)

# graphInfo = dot_to_nodes_and_edges("./scipy/classes_scipy.dot")
# graph = create_graph(graphInfo)
generate_graph_metric()
