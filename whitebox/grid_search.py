"""
  --------------------------------------------------
  File Name : grid_search.py
  Creation Date : 2019-06-27 N 10:37
  Last Modified : 2019-06-27 N 10:41 
  Created By : Joonatan Samuel
  --------------------------------------------------
"""


from sklearn.model_selection import cross_validate
from pprint import pprint

# ---- Choose a bunch of models ----
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neighbors


classifiers = {
    'Random Forest': sklearn.ensemble.RandomForestClassifier(),
    'Logistic Regression': sklearn.linear_model.LogisticRegression(),
    'Nearest Neighbors': sklearn.neighbors.KNeighborsClassifier()
    }

parameter_sets = {
    'Random Forest': [{'n_estimators': [1, 5, 10, 15, 25, 35],
                       'max_depth': [1, 2, 3, 5, 7, 10]}
                     ],
    'Logistic Regression': [{'penalty': ['l1', 'l2'],
                             'C': [0.1, 0.3, 1, 3, 10, 30, 100]}
                           ],
    # Very slow for some reason,
    # probably underlying implementation is slow
    #
    #'Support Vector Machine': [
    #                           {'kernel': ['linear'],
    #                                'C': [1, 10, 100, 1000]}
    #                          ],
    'Nearest Neighbors': [{'n_neighbors': range(1, 25, 3)}]

}

# TODO: rewrite this for loop to use this:
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
for name, model in classifiers.items():
    n_folds = 3
    scores = cross_validate(model, X, y, cv=n_folds, return_train_score=True)

    print("---- model {} ----".format(name))
    for fold in range(n_folds):
        print("Fold {} \t\t train score {:.2f}\t\t test score {:.2f}".format(
            fold,
            scores["train_score"][fold],
            scores["test_score"][fold]
        ))

    print()
