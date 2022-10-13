import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
from sklearn.metrics import classification_report
import pydotplus

# pd.set_option("display.max_columns", 100)


data = pd.read_csv("data/data_.csv")
data["Timestamp"] = pd.to_datetime(data["Timestamp"], format="%Y-%m-%d %H:%M:%S")

# 離群值篩選。正常值範圍：Q1 - 1.5IQR ~ Q3 + 1.5IQR
outlier = {}
for col in data.drop(columns=["Timestamp", "label"]).columns:
    Q1 = np.percentile(data[col], 25)
    Q3 = np.percentile(data[col], 75)
    IQR = Q3 - Q1
    outlier[col] = [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
for k, v in outlier.items():
    data = data[
        (data[k] > v[0]) &
        (data[k] < v[1])
    ]
data = data[(data["In_temperature"] > 0.0)]

data = data.drop(columns=["Timestamp", "thickness", "final thickness"])
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










