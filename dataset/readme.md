### 数据集来源
| project    | version        |
| ---------- | -------------- |
| tornado    | v4.2.0         |
| scipy      | v0.16.0b2      |
| ansible    | v1.9.2-0.1.rc1 |
| boto       | v2.38.0        |
| ipython    | rel-3.1.0      |
| numpy      | v1.9.2         |
| nltk       | v3.0.2         |
| matplotlib | v1.4.3         |
| django     | v1.8.2         |
| aws-cli    | v1.9.15        |
| Flashlight | v1.0.1         |
| Mailpile   | v0.5.2         |
| pyspider   | v0.3.6         |
| pyston     | v3.4.4         |
| reddit     |                |
| seaborn    | v0.6.0         |
| sentry     | v8.0.0-rc1     |
| yapf       | v0.6.1         |
| youtube-dl | v2015.12.31    |

### 工作流

#### 获取dot_file

**使用工具**

- pyreverse -- 绘制UML图

  ~~~sh
  pyreverse -p <project_name> -o dot ./<project_name>
  ~~~

- code2flow -- 绘制函数调用图

  ~~~sh
  code2flow -o dot --language py <sources>
  ~~~


#### 获取graph_node_metric

- 移动到getNodeMetric目录

- ~~~sh
  python ./detector.py
  ~~~

#### 获取graph_metric

- 分别移动到dot_file下的class与function文件夹下

- ~~~sh
  python ./generate_graph_metric.py
  ~~~

#### 获取train_dataset_metric_by_me

- 移动至graph_metric下

- ~~~sh
  python ./generate_train_dataset.py
  ~~~

#### 完成

至此已经获取所有训练数据，可以到machine_learning下进行训练。
