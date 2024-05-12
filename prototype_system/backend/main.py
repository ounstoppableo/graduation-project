from packages.code2flow import code2flow
from pylint.pyreverse.main import Run
from packages.getNodeMetric import detector
import os
import pandas as pd
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import json

class_smell_file = ['class_smell_metric.csv']
function_smell_file = ['function_smell_metric.csv']


def generate_dotfile (project_name,path):
    os.mkdir(f'./temp/dot_file/function/{project_name}')
    os.mkdir(f'./temp/dot_file/class/{project_name}')
    code2flow(path,f'./temp/dot_file/function/{project_name}/{project_name}_function_calls.dot','py',raw_source_dir=path)
    Run(['-p',project_name,'-o',"dot",'-d',f'./temp/dot_file/class/{project_name}',path])

def check_file_in_directory(directory, filename):
    # 检查指定的文件是否存在于目录中
    file_path = os.path.join(directory, filename)
    return os.path.exists(file_path)
def dot_to_nodes_and_edges_for_class(dot_file):
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

def create_diGraph_for_class(graphInfo:dict["nodes":any,"edges":any,"weights":any,"Instantiates":any]):
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

def generate_metric_for_class(graph) -> dict["degree":any,"neighbors":any,"in_degree":any,"out_degree":any,"self_loops":any]:
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

def generate_graph_metric_for_class(directory,target,project_name):
    subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir)) and subdir == project_name]
    for dirName in subdirectories:
        arr = []
        for fileName in os.listdir(os.path.join(directory,dirName)):
            graphInfo = dot_to_nodes_and_edges_for_class(os.path.join(directory,dirName,fileName))
            graph = create_diGraph_for_class(graphInfo)
            metrics = generate_metric_for_class(graph)
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


def dot_to_nodes_and_edges_for_function(dot_file):
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

def create_diGraph_for_function(graphInfo):
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

def generate_metric_for_function(graph) -> dict["degree":any,"neighbors_count":any,"in_degree":any,"out_degree":any,"self_loops":any,"max_call_circles":any]:
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

def generate_graph_metric_for_function(directory,target,project_name):
    subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir)) and subdir == project_name]
    for dirName in subdirectories:
        arr = []
        for fileName in os.listdir(os.path.join(directory,dirName)):
            graphInfo = dot_to_nodes_and_edges_for_function(os.path.join(directory,dirName,fileName))
            graph = create_diGraph_for_function(graphInfo)
            metrics = generate_metric_for_function(graph)
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
        

def concatGraphMetricFile(op_type,class_graph_metric_file_path='',function_graph_metric_file_path=''):
    graph_metric_arr = []
    files_and_folders = os.listdir(class_graph_metric_file_path if op_type=='class' else function_graph_metric_file_path)
    for fileName in files_and_folders:
            filePath = class_graph_metric_file_path+ '/' + fileName if op_type=='class' else function_graph_metric_file_path + '/' + fileName
            temp = pd.read_csv(filePath)
            columns = temp.columns.tolist()
            for item in temp.to_numpy().tolist():
                graph_metric_arr.append(item)

    graph_metric_arr = pd.DataFrame(graph_metric_arr,columns=columns)
    graph_metric_arr = graph_metric_arr.fillna(0)
    return graph_metric_arr.to_numpy().tolist()

def generate_class_smell_tag_dataset(project_name_index_in_path,output_path,class_graph_metric_file_path=''):
    for filePath in class_smell_file:
        graph_metric_arr = concatGraphMetricFile('class',class_graph_metric_file_path=class_graph_metric_file_path)
        train_dataset_columns = ['class_name','degree','neighbors_count','in_degree','out_degree','self_loops','nodes_inherit','nodes_inherited','nodes_aggregate','nodes_aggregated','nodes_combinate','nodes_combinated']
        train_dataset_arr=pd.DataFrame(graph_metric_arr,columns=train_dataset_columns)
        graph_node_metric = pd.read_csv(os.path.join(os.getcwd(),'.\\temp\\graph_node_metric\\class.csv'))
        new_columns = []
        for index, row in graph_node_metric.iterrows():
            temp = row["fileName"].split('\\')
            temp[-1] = temp[-1][:-3]
            temp = temp[project_name_index_in_path:]
            temp.append(str(row["className"]))
            new_columns.append('.'.join(temp))
        graph_node_metric['full_name'] = new_columns
        for index,row in train_dataset_arr.iterrows():
            filtered_df = graph_node_metric[graph_node_metric['full_name'].str.contains(row['class_name'])].reset_index(drop=True)
            try:
                train_dataset_arr.loc[index,'RCLOC'] = filtered_df.loc[0,'RCLOC']
                train_dataset_arr.loc[index,'CLOC'] = filtered_df.loc[0,'CLOC']
                train_dataset_arr.loc[index,'NOA'] = filtered_df.loc[0,'NOA']
                train_dataset_arr.loc[index,'NOM'] = filtered_df.loc[0,'NOM']
                train_dataset_arr.loc[index,'fileName'] = filtered_df.loc[0,'fileName']
                train_dataset_arr.loc[index,'lineno'] = filtered_df.loc[0,'lineno']
            except:
                train_dataset_arr.loc[index,'RCLOC'] = 0
                train_dataset_arr.loc[index,'CLOC'] = 0
                train_dataset_arr.loc[index,'NOA'] = 0
                train_dataset_arr.loc[index,'NOM'] = 0
                train_dataset_arr.loc[index,'fileName'] = -1
                train_dataset_arr.loc[index,'lineno'] = -1
            print(index)
        train_dataset_arr = train_dataset_arr[(train_dataset_arr['fileName'] != -1) & (train_dataset_arr['lineno'] != -1)]
        train_dataset_arr.to_csv(output_path+'/'+filePath,index=False)

def generate_function_smell_tag_dataset(project_name_index_in_path,output_path,function_graph_metric_file_path=''):
    for filePath in function_smell_file:
        graph_metric_arr = concatGraphMetricFile('function',function_graph_metric_file_path=function_graph_metric_file_path)
        train_dataset_columns = ['function_name','label','degree','neighbors_count','in_degree','out_degree','self_loops','max_call_circles']
        train_dataset_arr=pd.DataFrame(graph_metric_arr,columns=train_dataset_columns)
        graph_node_metric = pd.read_csv(os.path.join(os.getcwd(),'.\\temp\\graph_node_metric\\function.csv'))
        new_columns = []
        for index, row in graph_node_metric.iterrows():
            temp = row["fileName"].split('\\')
            temp[-1] = temp[-1][:-3]
            temp = temp[project_name_index_in_path:]
            temp = '.'.join(temp)
            new_columns.append(str(temp)+'.'+str(row["defName"]))
        graph_node_metric['full_name'] = new_columns
        for index,row in train_dataset_arr.iterrows():
            filtered_df = graph_node_metric[graph_node_metric['full_name'].str.contains(row['function_name'])].reset_index(drop=True)
            try:
                train_dataset_arr.loc[index,'RMLOC'] = filtered_df.loc[0,'RMLOC']
                train_dataset_arr.loc[index,'MLOC'] = filtered_df.loc[0,'MLOC']
                train_dataset_arr.loc[index,'PAR'] = filtered_df.loc[0,'PAR']
                train_dataset_arr.loc[index,'DOC'] = filtered_df.loc[0,'DOC']
                train_dataset_arr.loc[index,'fileName'] = filtered_df.loc[0,'fileName']
                train_dataset_arr.loc[index,'lineno'] = filtered_df.loc[0,'lineno']
            except:
                train_dataset_arr.loc[index,'RMLOC'] = 0
                train_dataset_arr.loc[index,'MLOC'] = 0
                train_dataset_arr.loc[index,'PAR'] = 0
                train_dataset_arr.loc[index,'DOC'] = 0
                train_dataset_arr.loc[index,'fileName'] = -1
                train_dataset_arr.loc[index,'lineno'] = -1
            print(index)
        train_dataset_arr = train_dataset_arr[(train_dataset_arr['fileName'] != -1) & (train_dataset_arr['lineno'] != -1)]
        train_dataset_arr.to_csv(output_path+'/'+filePath,index=False)

def pdToJson_class(raw):
    smells = ['large_class']
    res = {}
    for smell in smells:
        res[smell] = {}
        filePaths = raw.drop_duplicates(subset=['fileName'])['fileName'].to_numpy().tolist()
        for path in filePaths:
            res[smell][path] = []
        for index,row in raw.iterrows():
            if row[smell] == 1:
                res[smell][row.loc["fileName"]].append({
                "class": row.loc["class_name"],
                "start_line": row.loc["lineno"],
                "end_line": row.loc["RCLOC"] + row.loc["lineno"],
            })
    return res

def pdToJson_function(raw):
    smells = ['long_parameter_list','long_scope_chain','long_method']
    res = {}
    for smell in smells:
        res[smell] = {}
        filePaths = raw.drop_duplicates(subset=['fileName'])['fileName'].to_numpy().tolist()
        for path in filePaths:
            res[smell][path] = []
        for index,row in raw.iterrows():
            if row[smell] == 1:
                res[smell][row.loc["fileName"]].append({
                "function": row.loc["function_name"],
                "start_line": row.loc["lineno"],
                "end_line": row.loc["RMLOC"] + row.loc["lineno"],
            })
    return res

def smell_detection():
    with open('./model/LargeClass_rf.pkl', 'rb') as f:
        LargeClass_model = pickle.load(f)
    with open('./model/LongMethod_rf.pkl', 'rb') as f:
        LongMethod_model = pickle.load(f)
    with open('./model/LongParameterList_rf.pkl', 'rb') as f:
        LongParameterList_model = pickle.load(f)
    with open('./model/LongScopeChain_rf.pkl', 'rb') as f:
        LongScopeChain_model = pickle.load(f)

    class_metric = pd.read_csv('./temp/smell_metric/class_smell_metric.csv')
    function_metric = pd.read_csv('./temp/smell_metric/function_smell_metric.csv')
    class_metric_for_pred = class_metric.drop(['class_name','fileName','lineno','RCLOC'],axis=1)
    function_metric_for_pred = function_metric.drop(['function_name','label','fileName','lineno','RMLOC'],axis=1)
    

    LargeClass_pred_res = LargeClass_model.predict(class_metric_for_pred.to_numpy())
    class_metric["large_class"] = LargeClass_pred_res.tolist()
    class_res =  class_metric[class_metric["large_class"]==1]

    LongMethod_pred_res = LongMethod_model.predict(function_metric_for_pred.to_numpy())
    function_metric["long_method"] = LongMethod_pred_res.tolist()

    LongParameterList_pred_res = LongParameterList_model.predict(function_metric_for_pred.to_numpy())
    function_metric["long_parameter_list"] = LongParameterList_pred_res.tolist()

    LongScopeChain_pred_res = LongScopeChain_model.predict(function_metric_for_pred.to_numpy())
    function_metric["long_scope_chain"] = LongScopeChain_pred_res.tolist()

    function_res =  function_metric[(function_metric["long_scope_chain"]==1)|(function_metric["long_method"]==1)|(function_metric["long_parameter_list"]==1)]

    return {
        "class": pdToJson_class(class_res),
        "function": pdToJson_function(function_res),
    }

def delete_directory_contents(directory):
    # 遍历目录树中的所有文件和子文件夹
    for root, dirs, files in os.walk(directory, topdown=False):
        # 删除子文件
        for file_name in files:
            file_path = os.path.join(root, file_name)
            os.remove(file_path)
        # 删除子文件夹
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            os.rmdir(dir_path)

def parsing_file(path):
    if not (check_file_in_directory(path,'__init__.py')):
        raise ValueError('错误的包文件夹！！')
    project_name = path.split('\\')[-1]
    generate_dotfile (project_name,path)

def find_code_bad_smell(path):
    if not (check_file_in_directory(path,'__init__.py')):
        raise ValueError('错误的包文件夹！！')
    project_name = path.split('\\')[-1]
    project_name_index_in_path = path.split('\\').index(project_name)

    detector.run(path,'./temp/graph_node_metric')
    print("1.生成graph_node_metric成功！")

    generate_graph_metric_for_class(os.path.join(os.getcwd(),'.\\temp\\dot_file\\class'),os.path.join(os.getcwd(),'.\\temp\\graph_metric\\class\\'),project_name)
    delete_directory_contents(os.path.join(os.getcwd(),'.\\temp\\dot_file\\class'))

    generate_graph_metric_for_function(os.path.join(os.getcwd(),'.\\temp\\dot_file\\function'),os.path.join(os.getcwd(),'.\\temp\\graph_metric\\function\\'),project_name)
    delete_directory_contents(os.path.join(os.getcwd(),'.\\temp\\dot_file\\function'))
    print("2.生成graph_metric成功！")

    generate_class_smell_tag_dataset(project_name_index_in_path,os.path.join(os.getcwd(),'.\\temp\\smell_metric'),class_graph_metric_file_path=os.path.join(os.getcwd(),'.\\temp\\graph_metric\\class'))
    delete_directory_contents(os.path.join(os.getcwd(),'.\\temp\\graph_metric\\class'))
    
    generate_function_smell_tag_dataset(project_name_index_in_path,os.path.join(os.getcwd(),'.\\temp\\smell_metric'),function_graph_metric_file_path=os.path.join(os.getcwd(),'.\\temp\\graph_metric\\function'))
    delete_directory_contents(os.path.join(os.getcwd(),'.\\temp\\graph_metric\\function'))
    print("3.生成smell_metric成功！")

    result = smell_detection()
    print("4.检测完成！")
    return result

# try:
#     parsing_file('D:\\BaiduNetdiskDownload\\graduation_project\\project_folder\\ansible\\executor\\module_common.py')
# finally:

# find_code_bad_smell('D:\\BaiduNetdiskDownload\\graduation_project\\project_folder\\ansible')
