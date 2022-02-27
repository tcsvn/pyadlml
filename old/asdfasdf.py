import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# Reference for customizing plots : http://matplotlib.org/users/customizing.html
# print(plt.style.available)

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)
print("Classes {}".format(np.unique(y)))


#-----------------------------------
feature_names = data.feature_names
X_train = X_train
X_test = X_test
#-----------------------------------
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

models = {'gb':GradientBoostingClassifier(),
          'mlp':MLPClassifier(),
          'knn':KNeighborsClassifier(),
          'reg':LogisticRegression()}

for model_key in models:
    model = models[model_key]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    print("F1 for {0}: {1}".format(model_key, f1))


from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

interpreter = Interpretation(X_test, feature_names=feature_names)
model = InMemoryModel(models['knn'].predict_proba, examples=X_train)
fig, ax = interpreter.feature_importance.save_plot_feature_importance(
    model,
    ascending=True)
fig.show()
#model = InMemoryModel(models['knn'].predict_proba, examples=X_train)
#fig, ax = interpreter.feature_importance.save_plot_feature_importance(
#    model,
#    ascending=True)
#fig.show()
