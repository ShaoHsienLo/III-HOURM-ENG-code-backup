from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
from sklearn.metrics import classification_report
import pydotplus


data = pd.read_csv("data_.csv")

# 擷取時間段：8/13-8/14
data["Timestamp"] = pd.to_datetime(data["Timestamp"], format="%Y-%m-%d %H:%M:%S")
data = data[
    (data["Timestamp"] > pd.to_datetime("2022-08-13", format="%Y-%m-%d")) &
    (data["Timestamp"] < pd.to_datetime("2022-08-15", format="%Y-%m-%d"))
]
normal_data = data[data["label"] == "正常"]
abnormal_data = data[data["label"] == "異常"]
normal_rate = 0.82
normal_data = normal_data[:int(len(normal_data) * normal_rate)]
abnormal_data = abnormal_data[:int(len(abnormal_data) * (1 - normal_rate))]
data = pd.concat([normal_data, abnormal_data], axis=0)
print(data.shape)
print("正常數量: ", len(data[data["label"] == "正常"]))
print("異常數量: ", len(data[data["label"] == "異常"]))

data = data.drop(columns=["Timestamp", "thickness", "final thickness"])
print(data.columns)
exit(0)
X = data.drop(columns=["label"])
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))


d = {'feature': X.columns, 'importance': clf.feature_importances_}
p = pd.DataFrame(data=d).sort_values(by=['importance'], ascending=False)
print(p)

# graph = Source(tree.export_graphviz(clf, out_file=None, feature_names=X.columns))
# graph.format = 'png'
# graph.render('dtree_render', view=True)

# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=X.columns,
#                 class_names=["0", "1"])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('diabetes.png')
# Image(graph.create_png())










