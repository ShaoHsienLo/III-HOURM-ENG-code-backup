import random
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import export_graphviz
import pydot
from imblearn.combine import SMOTEENN
from sklearn.feature_selection import SelectKBest, f_classif
from lime.lime_tabular import LimeTabularExplainer
from loguru import logger
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample

pd.set_option('display.max_columns', 100)


def read_file(path, file):
    logger.info("read_file ...")
    df = pd.read_csv(os.path.join(path, file))
    return df


def handle_na_values(df):
    logger.info("handle na values ...")
    df_handle_na = df.fillna(method="bfill")
    return df_handle_na


def handle_categorical_data(df_handle_na, label="label"):
    logger.info("handle categorical data ...")
    labelencoder = LabelEncoder()
    df_handle_na[label] = labelencoder.fit_transform(df_handle_na[label])
    df_encoded = df_handle_na.copy()
    target_names = list(labelencoder.classes_)
    mapping = {}
    for cl in labelencoder.classes_:
        mapping[cl] = labelencoder.transform([cl])[0]
    print(mapping)
    return df_encoded, target_names


def process_outlier(df_handle_na, label="label"):
    logger.info("process outlier ...")
    outlier = {}
    for col in df_handle_na.drop(columns=[label]).columns:
        Q1 = np.percentile(df_handle_na[col], 25)
        Q3 = np.percentile(df_handle_na[col], 75)
        IQR = Q3 - Q1
        outlier[col] = [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
    for k, v in outlier.items():
        df_handle_na = df_handle_na[
            (df_handle_na[k] > v[0]) &
            (df_handle_na[k] < v[1])
        ]
    df_no_outlier = df_handle_na[(df_handle_na["In_temperature"] > 0.0)]
    print(df_no_outlier.describe())
    return df_no_outlier


def select_k_best(X_train, y_train, X_test, k=2):
    logger.info("select k best ...")
    fs = SelectKBest(score_func=f_classif, k=k)
    fs.fit(X_train, y_train)
    features = X_train.columns
    features_selected = features[fs.get_support()]
    # print("p value:")
    # print(sorted(fs.pvalues_))
    # print("new features:")
    # print(features[fs.get_support()])

    X_train_selected = X_train[features_selected]
    X_test_selected = X_test[features_selected]
    return X_train_selected, X_test_selected


def split_data(df, label="label"):
    logger.info("split data ...")
    random_state = random.randint(0, 100)
    X = df.drop(columns=[label])
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    print("Training set size: ", len(X_train))
    print("Testing set size: ", len(X_test))
    print("Total data size: ", len(X))
    print("Labels value counts:\n", y.value_counts())
    return X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def normalization(X_train, X_test):
    logger.info("normalization ...")
    scale = MinMaxScaler()
    X_train_norm = pd.DataFrame(scale.fit_transform(X_train), columns=X_train.columns)
    X_test_norm = pd.DataFrame(scale.transform(X_test), columns=X_test.columns)
    return X_train_norm, X_test_norm


def smoteenn(X_train_norm, y_train):
    logger.info("smoteenn ...")
    sm = SMOTEENN()
    X_train_res, y_train_res = sm.fit_sample(X_train_norm, y_train)
    # print("X_train_res:\n", X_train_res)
    # print("y_train_res:\n", y_train_res)
    return X_train_res, y_train_res


def resampling(X_train_norm, y_train, method="upsample"):
    logger.info("resample ...")
    random_state = random.randint(0, 100)
    X = pd.concat([X_train_norm, y_train], axis=1)
    X.columns = list(X_train_norm.columns) + ["label"]
    X_train_normal = X[X["label"] == 0]
    X_train_abnormal = X[X["label"] == 1]
    X_res = pd.DataFrame()

    if method == "upsample":
        X_train_abnormal_upsampled = resample(X_train_abnormal, random_state=random_state,
                                              n_samples=len(X_train_normal), replace=True)
        X_res = pd.concat([X_train_normal, X_train_abnormal_upsampled])
    elif method == "downsample":
        X_train_normal_downsampled = resample(X_train_normal, random_state=random_state,
                                              n_samples=len(X_train_abnormal), replace=True)
        X_res = pd.concat([X_train_normal_downsampled, X_train_abnormal])
    print("Labels value counts(After resample):")
    print(X_res["label"].value_counts())
    return X_res.drop(columns=["label"]), X_res["label"]


class ModelTraining:

    def __init__(self, X_train_res, y_train_res):
        self.X_train_res = X_train_res
        self.y_train_res = y_train_res

    def rf(self):
        logger.info("rf model training ...")
        rf = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=3)
        rf.fit(self.X_train_res, self.y_train_res)
        return rf

    def knn(self):
        logger.info("knn model training ...")
        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(self.X_train_res, self.y_train_res)
        return knn
        # predicted = knn.predict(X_train)
        # print('訓練集: ', knn.score(X_train_res, y_train_res))
        # print('測試集: ', knn.score(X_test, y_test))

    def dt(self):
        logger.info("dt model training ...")
        dt = tree.DecisionTreeClassifier()
        dt.fit(self.X_train_res, self.y_train_res)
        return dt

    def lr(self):
        logger.info("lr model training ...")
        lr = LogisticRegression(solver="liblinear")  # lbfgs, liblinear
        lr.fit(self.X_train_res, self.y_train_res)
        return lr


def explain_model(rf, X_train, X_test, target_names=None, model_name="rf.model"):
    logger.info("explain model ...")
    X_train_ = X_train.copy().reset_index(drop=True)
    X_test_ = X_test.copy().reset_index(drop=True)
    explainer = LimeTabularExplainer(X_train_.values, feature_names=X_train_.columns, class_names=target_names)
    i = random.randint(0, X_test_.shape[0])
    exp = explainer.explain_instance(X_test_.iloc[i], rf.predict_proba)
    exp.save_to_file("./explain-model/20221024/{}-exp.html".format(model_name))


def show_performances(model, X_test_norm, y_test, target_names=None):
    logger.info("show performances ...")
    y_pred = model.predict(X_test_norm)

    try:
        print("Confusion metric:\n{}\n".format(classification_report(y_test, y_pred, target_names=target_names)))
    except Exception as e:
        print(e)

    try:
        importrances = {'feature': X_test_norm.columns, 'importance': model.feature_importances_}
        importrances_df = pd.DataFrame(data=importrances).sort_values(by=['importance'], ascending=False)
        print("Feature importances:\n{}\n".format(importrances_df))
    except Exception as e:
        print(e)

    try:
        score = roc_auc_score(y_test, model.predict_proba(X_test_norm)[:, 1])
        print("ROC SUC score:\n{}\n".format(score))
    except Exception as e:
        print(e)


def visualization(rf, feature_list):
    logger.info("visualization ...")
    tree = rf.estimators_[5]
    export_graphviz(tree, out_file='./visualization/tree.dot', feature_names=feature_list, rounded=True, precision=1)
    (graph,) = pydot.graph_from_dot_file('./visualization/tree.dot')
    graph.write_png('./visualization/tree.png')


def save_model(model, model_name="rf.model"):
    logger.info("save model ...")
    joblib.dump(model, "model-data/20221024/{}".format(model_name))


def load_model(model_name="rf.model"):
    logger.info("load model ...")
    model = joblib.load("model-data/20221024/{}".format(model_name))
    return model


if __name__ == "__main__":
    # 讀檔
    df = read_file(r"C:\Users\samuello\Downloads\III\宏英\code\data", "data.csv")
    # df = df.drop(columns=["Timestamp", "thickness", "final thickness"])

    # 處理遺失值
    df_handle_na = handle_na_values(df)

    # 處理類別資料
    df_encoded, target_names = handle_categorical_data(df_handle_na)

    # 處理離群值...
    df_no_outlier = df_encoded
    # df_no_outlier = process_outlier(df_encoded)

    # 切分模型輸入資料與預測目標
    X_train, X_test, y_train, y_test = split_data(df_no_outlier)

    # 特徵選擇(降維)
    X_train_selected, X_test_selected = select_k_best(X_train, y_train, X_test, k=len(X_train.columns))

    # 資料縮放
    X_train_norm, X_test_norm = normalization(X_train_selected, X_test_selected)

    # 處理資料不平衡
    # X_train_res = X_train_norm
    # y_train_res = y_train
    # X_train_res, y_train_res = smoteenn(X_train_norm, y_train)
    X_train_res, y_train_res = resampling(X_train_norm, y_train, method="downsample")

    # 模型訓練
    model_training = ModelTraining(X_train_res, y_train_res)
    models = [model_training.rf(), model_training.knn(), model_training.dt(), model_training.lr()]
    # models = [model_training.rf()]
    model_names = ["rf.model", "knn.model", "dt.model", "lr.model"]
    # model_names = ["rf.model"]

    for model, model_name in zip(models, model_names):
        # 解釋模型
        explain_model(model, X_train_res, X_test_norm, target_names=target_names, model_name=model_name)

        # 儲存模型
        print(model_name)
        # save_model(model, model_name=model_name)

        # 載入模型
        model = load_model(model_name=model_name)

        # 印出模型效能數據
        show_performances(model, X_test_norm, y_test, target_names=target_names)

        # 輸出決策樹圖
        # visualization(rf, X_train.columns)
