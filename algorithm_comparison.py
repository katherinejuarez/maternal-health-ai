#source: https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/

# Compare Algorithms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# load dataset
dataframe = pd.read_csv('data.csv')
features = dataframe.iloc[:, :-1].values
labels = dataframe['RiskLevel']

array = dataframe.values
X = array[:,0:6]
Y = array[:,6]
# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('SVM', SVC()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))

# evaluate each model in turn
results = []
labels = []
scoring = 'accuracy'
for label, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    labels.append(label)
    msg = "%s: %f (%f)" % (label, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(labels)
plt.show()