from sklearn.datasets import load_iris
from skrules import SkopeRules
import numpy as np
"""
dataset = load_iris()
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
clf = SkopeRules(max_depth_duplication=2,
                 n_estimators=30,
                 precision_min=0.3,
                 recall_min=0.1,
                 feature_names=feature_names)

for idx, species in enumerate(dataset.target_names):
    X, y = dataset.data, dataset.target
    clf.fit(X, y == idx)
    rules = clf.rules_[0:3]
    print("Rules for iris", species)
    for rule in rules:
        print(rule)
    print()
    print(20*'=')
    print()
"""
clf = SkopeRules(max_depth_duplication=2,
                 n_estimators=30,
                 precision_min=0.3,
                 recall_min=0.1,
                 )


data = np.array(
    [np.array([1, 1, 1, 0, 1, 1, 1, 1]),
     np.array([0, 1, 1, 0, 1, 1, 1, 1]),
     np.array([0, 1, 1, 0, 1, 1, 1, 1]),
     np.array([1, 1, 1, 1, 1, 1, 1, 1])])
label = np.array([1,
                  0,
                  0,
                  0
                  ])

clf.fit(data,label)
rules = clf.rules_[0:3]
for rule in rules:
    print(rule)