# classification_models.py

# Importing necessary libraries
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# List of classification models
logistic_regression = LogisticRegression()
decision_tree_classifier = DecisionTreeClassifier()
random_forest_classifier = RandomForestClassifier()
gradient_boosting_classifier = GradientBoostingClassifier()
svc = SVC(probability=True)  # Note: probability=True for getting class probabilities
knn_classifier = KNeighborsClassifier()
naive_bayes = GaussianNB()
mlp_classifier = MLPClassifier()
xgb_classifier = XGBClassifier()
lgbm_classifier = LGBMClassifier()
catboost_classifier = CatBoostClassifier()

# You can add more models as needed

if __name__ == "__main__":
    # You can add some code here to test or use the models
    pass
