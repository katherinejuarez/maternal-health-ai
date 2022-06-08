import pandas as pd
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar


def main():
    data = pd.read_csv('data.csv')
    print(data.head(5))

    # UPDATE ME:: use this line to remove certain columns
    # data.drop('Age', inplace=True, axis=1)

    features = data.iloc[:, :-1].values
    labels = data['RiskLevel']

    scaled = preprocessing.normalize(features)

    # define classifiers
    svc_model = SVC()
    knn_model = KNeighborsClassifier()
    gnb_model = GaussianNB()
    log_model = LogisticRegression()
    tree_model = DecisionTreeClassifier()
    forest_model = RandomForestClassifier()

    # UPDATE ME:: choose classifier1 and classifier2 here
    classifier1 = tree_model
    classifier1.fit(features, labels)

    cv = KFold(n_splits=10, shuffle=True, random_state=None)
    scores = cross_val_score(classifier1, scaled, labels, cv=cv, scoring='accuracy')

    output1 = classifier1.predict(features)
    print(output1)

    classifier2 = forest_model
    classifier2.fit(features, labels)

    cv = KFold(n_splits=10, shuffle=True, random_state=None)
    scores = cross_val_score(classifier2, scaled, labels, cv=cv, scoring='accuracy')

    output2 = classifier2.predict(features)
    print(output2)

    tb = mcnemar_table(y_target=labels,
                       y_model1=output1,
                       y_model2=output2)
    print(tb)

    chi2, p = mcnemar(ary=tb, corrected=True)
    print('chi-squared:', chi2)
    print('p-value:', p)


if __name__ == '__main__':
    main()
