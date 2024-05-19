from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection  import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasClassifier
import pickle
from sklearn.ensemble import VotingClassifier

def np_arr_to_fit_cnn_input(np_arr,a,b,c):
    arr = []
    for row in np_arr:
        temp = []
        for piece in np.hsplit(np.pad(row, (0, a*b*c-len(row)), 'constant', constant_values=0),a):
            temp.append(np.reshape(piece,(b,c)).tolist())
        arr.append(temp)
    return np.array(arr)

# def calculateF1(precision,recall):
#     return 2 * precision * recall / (precision + recall)

def model_train_function(smell_type,X,y,cnn_2d_w=4,cnn_2d_h=4):
    X = X.astype(float)
    y = y.astype(float)
    X_train, x_test, Y_train, y_test = train_test_split(X,y,test_size=0.25)
    rfolds = RepeatedStratifiedKFold(n_repeats=4)
    #随机森林
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, Y_train)
    y_rf_pred = rf_model.predict(x_test)
    rf_result = np.column_stack((x_test, y_rf_pred)).tolist()
    with open(f'./models/{smell_type}_rf.pkl', 'wb') as file1:
        pickle.dump(rf_model, file1)
    rf_precision_scores =  cross_val_score(rf_model,X,y,cv=rfolds,scoring='precision')
    rf_recall_scores =  cross_val_score(rf_model,X,y,cv=rfolds,scoring='recall')
    rf_f1_scores =  cross_val_score(rf_model,X,y,cv=rfolds,scoring='f1')

    print("-------------------------------------RandomForest----------------------------------------")
    print(f'precision: {rf_precision_scores.mean()}')
    print(f'recall: {rf_recall_scores.mean()}')
    print(f'f-score: {rf_f1_scores.mean()}')

    #支持向量机
    svc_model = SVC(kernel='linear', random_state=42)
    svc_model.fit(X_train, Y_train)
    y_svc_pred = svc_model.predict(x_test)
    with open(f'./models/{smell_type}_svc.pkl', 'wb') as file2:
        pickle.dump(svc_model, file2)
    svc_result = np.column_stack((x_test, y_svc_pred)).tolist()
    svc_precision_scores =  cross_val_score(svc_model,X,y,cv=rfolds,scoring='precision')
    svc_recall_scores =  cross_val_score(svc_model,X,y,cv=rfolds,scoring='recall')
    svc_f1_scores =  cross_val_score(svc_model,X,y,cv=rfolds,scoring='f1')

    print("---------------------------------------SVC--------------------------------------")
    print(f'precision: {svc_precision_scores.mean()}')
    print(f'recall: {svc_recall_scores.mean()}')
    print(f'f-score: {svc_f1_scores.mean()}')

    #朴素贝叶斯
    nb_model = GaussianNB()
    nb_model.fit(X_train, Y_train)
    y_nb_pred = nb_model.predict(x_test)
    with open(f'./models/{smell_type}_nb.pkl', 'wb') as file3:
        pickle.dump(nb_model, file3)
    nb_result = np.column_stack((x_test, y_nb_pred)).tolist()
    nb_precision_scores =  cross_val_score(nb_model,X,y,cv=rfolds,scoring='precision')
    nb_recall_scores =  cross_val_score(nb_model,X,y,cv=rfolds,scoring='recall')
    nb_f1_scores =  cross_val_score(nb_model,X,y,cv=rfolds,scoring='f1')

    print("-------------------------------------NB----------------------------------------")
    print(f'precision: {nb_precision_scores.mean()}')
    print(f'recall: {nb_recall_scores.mean()}')
    print(f'f-score: {nb_f1_scores.mean()}')

    # 逻辑回归
    lr_model = LogisticRegression()
    lr_model.fit(X_train, Y_train)
    y_lr_pred = lr_model.predict(x_test)
    with open(f'./models/{smell_type}_lr.pkl', 'wb') as file3:
        pickle.dump(lr_model, file3)
    lr_result = np.column_stack((x_test, y_lr_pred)).tolist()
    lr_precision_scores =  cross_val_score(lr_model,X,y,cv=rfolds,scoring='precision')
    lr_recall_scores =  cross_val_score(lr_model,X,y,cv=rfolds,scoring='recall')
    lr_f1_scores =  cross_val_score(lr_model,X,y,cv=rfolds,scoring='f1')

    print("-------------------------------------LR----------------------------------------")
    print(f'precision: {lr_precision_scores.mean()}')
    print(f'recall: {lr_recall_scores.mean()}')
    print(f'f-score: {lr_f1_scores.mean()}')


    # 构建 CNN 模型
    X_for_cnn = np_arr_to_fit_cnn_input(X.copy(),4,4,1)
    X_train_for_cnn = np_arr_to_fit_cnn_input(X_train.copy(),4,4,1)
    x_test_for_cnn = np_arr_to_fit_cnn_input(x_test.copy(),4,4,1)
    Y_train_for_cnn = Y_train.copy().astype(float)
    y_test_for_cnn = y_test.copy().astype(float)
    cnn_model_1D = keras.Sequential([
        keras.layers.Conv1D(16, kernel_size=(2), activation='relu', input_shape=(X_train[0].shape[0],1)),
        keras.layers.MaxPooling1D(pool_size=(1)),
        keras.layers.Conv1D(32, kernel_size=(2), activation='relu'),
        keras.layers.MaxPooling1D(pool_size=(1)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    cnn_model_2D = keras.Sequential([
        keras.layers.Conv2D(16, kernel_size=(2, 2), activation='relu', input_shape=(cnn_2d_w, cnn_2d_h, 1)),
        keras.layers.MaxPooling2D(pool_size=(1, 1)),
        keras.layers.Conv2D(32, kernel_size=(2, 2), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(1, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # 编译模型
    cnn_model_1D.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model_2D.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model_1D.save(f'models/{smell_type}_cnn_1D.h5')
    cnn_model_2D.save(f'models/{smell_type}cnn_2D.h5')

    #转换成scikit-learn 模型
    cnn_1D_clf = KerasClassifier(build_fn=cnn_model_1D, epochs=10,batch_size=10,verbose=0)
    cnn_2D_clf = KerasClassifier(build_fn=cnn_model_2D, epochs=10,batch_size=10,verbose=0)

    #训练
    cnn_model_1D.fit(X_train, Y_train_for_cnn, epochs=10,batch_size=10,validation_data=(x_test, y_test_for_cnn),verbose=0)
    cnn_model_2D.fit(X_train_for_cnn, Y_train_for_cnn, epochs=10,batch_size=10,validation_data=(x_test_for_cnn, y_test_for_cnn),verbose=0)

    # 进行预测
    y_cnn_pred_1D = cnn_model_1D.predict(x_test)
    cnn_1D_result = np.column_stack((x_test, np.round(y_cnn_pred_1D.flatten(),decimals=0))).tolist()
    y_cnn_pred_2D = cnn_model_2D.predict(x_test_for_cnn)
    cnn_2D_result = np.column_stack((x_test, np.round(y_cnn_pred_2D.flatten(),decimals=0))).tolist()

    # 分数计算
    cnn_1D_precision_scores =  cross_val_score(cnn_1D_clf,X,y,cv=rfolds,scoring='precision')
    cnn_1D_recall_scores =  cross_val_score(cnn_1D_clf,X,y,cv=rfolds,scoring='recall')
    cnn_1D_f1_scores =  cross_val_score(cnn_1D_clf,X,y,cv=rfolds,scoring='f1')

    cnn_2D_precision_scores =  cross_val_score(cnn_2D_clf,X_for_cnn,y,cv=rfolds,scoring='precision')
    cnn_2D_recall_scores =  cross_val_score(cnn_2D_clf,X_for_cnn,y,cv=rfolds,scoring='recall')
    cnn_2D_f1_scores =  cross_val_score(cnn_2D_clf,X_for_cnn,y,cv=rfolds,scoring='f1')


    # 输出分类报告
    print("-------------------------------------CNN----------------------------------------")
    print("1维卷积:")
    print(f'precision: {cnn_1D_precision_scores.mean()}')
    print(f'recall: {cnn_1D_recall_scores.mean()}')
    print(f'f-score: {cnn_1D_f1_scores.mean()}')

    print("2维卷积:")
    print(f'precision: {cnn_2D_precision_scores.mean()}')
    print(f'recall: {cnn_2D_recall_scores.mean()}')
    print(f'f-score: {cnn_2D_f1_scores.mean()}')


    # 构建Bagging模型，添加多个基本模型，并进行交叉验证
    bagging_model = VotingClassifier(estimators=[('rf',rf_model),('svc',svc_model),('nb',nb_model),('lr',lr_model),('cnn',cnn_1D_clf)], voting='hard')
    bagging_model.fit(X_train, Y_train)
    bagging_pred = bagging_model.predict(x_test)
    with open(f'./models/{smell_type}_bagging.pkl', 'wb') as file7:
        pickle.dump(bagging_model, file7)
    bagging_result = np.column_stack((x_test, bagging_pred)).tolist()
    bagging_precision_scores =  cross_val_score(bagging_model,X,y,cv=rfolds,scoring='precision')
    bagging_recall_scores =  cross_val_score(bagging_model,X,y,cv=rfolds,scoring='recall')
    bagging_f1_scores =  cross_val_score(bagging_model,X,y,cv=rfolds,scoring='f1')
    # 输出分类报告
    print("-------------------------------------BAGGING----------------------------------------")
    print(f'precision: {bagging_precision_scores.mean()}')
    print(f'recall: {bagging_recall_scores.mean()}')
    print(f'f-score: {bagging_f1_scores.mean()}')

    
    df=pd.DataFrame({'RF':rf_precision_scores,'SVC':svc_precision_scores,'NB':nb_precision_scores,'LR':lr_precision_scores
                ,'CNN':cnn_1D_precision_scores,'CNN_2D':cnn_2D_precision_scores,'bagging':bagging_precision_scores})
    df = df.drop(columns=['CNN_2D'])
    boxplot = sns.boxplot(x="variable", y="value", data=pd.melt(df),palette="Pastel1")
    boxplot.axes.set_title(smell_type, fontsize=14)
    boxplot.set_xlabel("Classifier", fontsize=14)
    boxplot.set_ylabel("precision", fontsize=14)
    plt.show()

    df2=pd.DataFrame({'RF':rf_recall_scores,'SVC':svc_recall_scores,'NB':nb_recall_scores,'LR':lr_recall_scores
                ,'CNN':cnn_1D_recall_scores,'CNN_2D':cnn_2D_recall_scores,'bagging':bagging_recall_scores})
    df2 = df2.drop(columns=['CNN_2D'])
    boxplot2 = sns.boxplot(x="variable", y="value", data=pd.melt(df2),palette="Pastel1")
    boxplot2.axes.set_title(smell_type, fontsize=14)
    boxplot2.set_xlabel("Classifier", fontsize=14)
    boxplot2.set_ylabel("recall", fontsize=14)
    plt.show()

    df3=pd.DataFrame({'RF':rf_f1_scores,'SVC':svc_f1_scores,'NB':nb_f1_scores,'LR':lr_f1_scores
                ,'CNN':cnn_1D_f1_scores,'CNN_2D':cnn_2D_f1_scores,'bagging':bagging_f1_scores})
    df3 = df3.drop(columns=['CNN_2D'])
    boxplot3 = sns.boxplot(x="variable", y="value", data=pd.melt(df3),palette="Pastel1")
    boxplot3.axes.set_title(smell_type, fontsize=14)
    boxplot3.set_xlabel("Classifier", fontsize=14)
    boxplot3.set_ylabel("f-score", fontsize=14)
    plt.show()

    return {
        "rf": {
            "precision": rf_precision_scores.mean(),
            "recall": rf_recall_scores.mean(),
            "f1": rf_f1_scores.mean(),
        },
        "svc": {
            "precision": svc_precision_scores.mean(),
            "recall": svc_recall_scores.mean(),
            "f1": svc_f1_scores.mean(),
        },
        "nb": {
            "precision": nb_precision_scores.mean(),
            "recall": nb_recall_scores.mean(),
            "f1": nb_f1_scores.mean(),
        },
        "lr": {
            "precision": lr_precision_scores.mean(),
            "recall": lr_recall_scores.mean(),
            "f1": lr_f1_scores.mean(),
        },
        "cnn_1D": {
            "precision": cnn_1D_precision_scores.mean(),
            "recall": cnn_1D_recall_scores.mean(),
            "f1": cnn_1D_f1_scores.mean(),
        },
        "cnn_2D": {
            "precision": cnn_2D_precision_scores.mean(),
            "recall": cnn_2D_recall_scores.mean(),
            "f1": cnn_2D_f1_scores.mean(),
        },
        "bagging": {
            "precision": bagging_precision_scores.mean(),
            "recall": bagging_recall_scores.mean(),
            "f1": bagging_f1_scores.mean(),
        }
    },{
        'rf_result':rf_result,
        'svc_result':svc_result,
        'nb_result':nb_result,
        'lr_result':lr_result,
        'cnn_1D_result':cnn_1D_result,
        'cnn_2D_result':cnn_2D_result,
        'bagging_result':bagging_result,
    }


