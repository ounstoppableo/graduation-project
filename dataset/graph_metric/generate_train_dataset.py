import pandas as pd
import os
import numpy as np
class_smell_file = ['LargeClass_YesNo.csv']
function_smell_file = ['LongMethod_YesNo.csv','LongMethod_YesNo.csv','LongParameterList_YesNo.csv',
'LongScopeChaining_YesNo.csv']

class_graph_metric_file_path = './class'
function_graph_metric_file_path = './function'
train_target_path = '../train_dataset_metric_by_me'

def concatGraphMetricFile(op_type):
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
    return graph_metric_arr.to_numpy().tolist(),columns


def generate_class_smell_tag_dataset():
    columns_to_drop = ["AvgCyclomatic","AvgCyclomaticModified","AvgCyclomaticStrict","AvgEssential","AvgLine","AvgLineBlank","AvgLineCode","AvgLineComment","CountClassBase","CountClassCoupled","CountClassCoupledModified","CountClassDerived","CountDeclInstanceMethod","CountDeclInstanceVariable","CountDeclMethod","CountDeclMethodAll","CountLine","CountLineBlank","CountLineCode","CountLineCodeDecl","CountLineCodeExe","CountLineComment","CountStmt","CountStmtDecl","CountStmtExe","MaxCyclomatic","MaxCyclomaticModified","MaxCyclomaticStrict","MaxEssential","MaxInheritanceTree","MaxNesting","RatioCommentToCode","SumCyclomatic","SumCyclomaticModified","SumCyclomaticStrict","SumEssential","number of parameter","number of parameter without *args, **kargs","cohesion"]
    for filePath in class_smell_file:
        tag_df = pd.read_csv('../origin_tag_dataset/'+filePath)
        tag_df = tag_df.drop(columns_to_drop, axis=1)

        graph_metric_arr,columns = concatGraphMetricFile('class')
        graph_metric_df = pd.DataFrame(graph_metric_arr,columns=columns)
        train_dataset_arr = []
        train_dataset_columns = ['Kind','lineno','class_name','degree','neighbors_count','in_degree','out_degree','self_loops','nodes_inherit','nodes_inherited','nodes_aggregate','nodes_aggregated','nodes_combinate','nodes_combinated','smell']
        for index,row in tag_df.iterrows():
            temp = graph_metric_df[graph_metric_df['class_name']==row['understandname']].to_numpy().tolist()
            if(len(temp)!=0):
                train_data_row = [row["Kind"],row["lineno"]]
                for metric in temp[0]:
                    train_data_row.append(metric)
                train_data_row.append(row["smell"])
                train_dataset_arr.append(train_data_row)
        train_dataset_arr=pd.DataFrame(train_dataset_arr,columns=train_dataset_columns)
        graph_node_metric = pd.read_csv('../getNodeMetric/graph_node_metric/class.csv')
        new_columns = []
        for index, row in graph_node_metric.iterrows():
            temp = row["fileName"].split('\\')
            temp[-1] = temp[-1][:-3]
            temp = temp[2:]
            temp.append(str(row["className"]))
            new_columns.append('.'.join(temp))
        graph_node_metric['full_name'] = new_columns
        for index,row in train_dataset_arr.iterrows():
            filtered_df = graph_node_metric[graph_node_metric['full_name'].str.contains(row['class_name'])].reset_index(drop=True)
            try:
                train_dataset_arr.loc[index,'CLOC'] = filtered_df.loc[0,'CLOC']
                train_dataset_arr.loc[index,'NOA'] = filtered_df.loc[0,'NOA']
                train_dataset_arr.loc[index,'NOM'] = filtered_df.loc[0,'NOM']
            except:
                train_dataset_arr.loc[index,'CLOC'] = 0
                train_dataset_arr.loc[index,'NOA'] = 0
                train_dataset_arr.loc[index,'NOM'] = 0
            print(index)
        train_dataset_arr["smell"] = train_dataset_arr.pop("smell")
        train_dataset_arr.to_csv(train_target_path+'/'+filePath,index=False)
        
def generate_function_smell_tag_dataset():
    columns_to_drop = ['CountLine','CountLineBlank','CountLineCode','CountLineCodeDecl','CountLineCodeExe','CountLineComment','CountPath','CountPathLog','CountStmt','CountStmtDecl','CountStmtExe','Cyclomatic','CyclomaticModified','CyclomaticStrict','Essential','MaxNesting','RatioCommentToCode','number of parameter','number of parameter without *args, **kargs']
    for filePath in function_smell_file:
        tag_df = pd.read_csv('../origin_tag_dataset/'+filePath)
        tag_df = tag_df.drop(columns_to_drop, axis=1)

        graph_metric_arr,columns = graph_metric_arr,columns = concatGraphMetricFile('function')
        graph_metric_df = pd.DataFrame(graph_metric_arr,columns=columns)
        for index, row in graph_metric_df.iterrows():
            temp = '.'.join(graph_metric_df.loc[index,'full_name'].split('::'))
            print(temp)
            graph_metric_df.loc[index,'full_name'] = temp
        train_dataset_arr = []
        train_dataset_columns = ['Kind','function_name','lineno','degree','neighbors_count','in_degree','out_degree','self_loops','max_call_circles','smell']
        for index,row in tag_df.iterrows():
            understandname = row['understandname'].strip('\"\"').split('(')[0]
            print(understandname)
            temp = graph_metric_df[graph_metric_df['full_name']==understandname].to_numpy().tolist()
            if(len(temp)!=0):
                train_data_row = [row["Kind"]]
                for metric in temp[0]:
                    train_data_row.append(metric)
                train_data_row.append(row["smell"])
                train_dataset_arr.append(train_data_row)
        train_dataset_arr=pd.DataFrame(train_dataset_arr,columns=train_dataset_columns)
        graph_node_metric = pd.read_csv('../getNodeMetric/graph_node_metric/function.csv')
        new_columns = []
        for index, row in graph_node_metric.iterrows():
            temp = row["fileName"].split('\\')
            temp[-1] = temp[-1][:-3]
            temp = temp[2:]
            temp = '.'.join(temp)
            new_columns.append(str(temp)+'.'+str(row["defName"]))
        graph_node_metric['full_name'] = new_columns
        for index,row in train_dataset_arr.iterrows():
            filtered_df = graph_node_metric[graph_node_metric['full_name'].str.contains(row['function_name'])].reset_index(drop=True)
            try:
                train_dataset_arr.loc[index,'MLOC'] = filtered_df.loc[0,'MLOC']
                train_dataset_arr.loc[index,'PAR'] = filtered_df.loc[0,'PAR']
                train_dataset_arr.loc[index,'DOC'] = filtered_df.loc[0,'DOC']
            except:
                train_dataset_arr.loc[index,'MLOC'] = 0
                train_dataset_arr.loc[index,'PAR'] = 0
                train_dataset_arr.loc[index,'DOC'] = 0
            print(index)
        train_dataset_arr["smell"] = train_dataset_arr.pop("smell")
        train_dataset_arr.to_csv(train_target_path+'/'+filePath,index=False)

# concatGraphMetricFile('function')
generate_class_smell_tag_dataset()
generate_function_smell_tag_dataset()