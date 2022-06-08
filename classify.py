import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



def main():
    data = pd.read_csv('data.csv')
    data.dropna(inplace=True)

    features = data[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]

    # features = data[['SystolicBP', 'BS']]

    # print(features)
    labels = data['RiskLevel']

    ss = preprocessing.StandardScaler()
    scaled = ss.fit_transform(features)

    # define classifiers
    svc_model = SVC()
    knn_model = KNeighborsClassifier()
    gnb_model = GaussianNB()
    log_model = LogisticRegression()
    tree_model = DecisionTreeClassifier()
    forest_model = RandomForestClassifier()

    # UPDATE ME:: choose classifier here
    classifier = forest_model

    # cv = ShuffleSplit(n_splits=20, test_size=0.3, random_state=42)
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(classifier, scaled, labels, cv=cv, scoring='accuracy')

    print("accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # UPDATE ME:: choose classifier1 here to the same as classifier
    classifier1 = svc_model
    classifier1.fit(features, labels)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(classifier1, scaled, labels, cv=cv, scoring='accuracy')

    output = classifier1.predict(features)

    high=0
    mid=0
    low=0

    for l in labels:
        if l=="high risk":
            high+=1
        if l=="mid risk":
            mid+=1
        if l=="low risk":
            low+=1

    # print(high, mid, low)


    #calculate false positives and false negatives for high risk pregnancies
    
    FP_high=0
    FN_high=0

    for l, s in zip(labels, output):
        if l!="high risk" and s=="high risk":
            FP_high+=1
        if l=="high risk" and s!="high risk":
            FN_high+=1

    print("FP_high: ", FP_high)
    print("FN_high: ", FN_high)




    #calculate accuracies for different risk levels -- high, mid, low
    

    num=0
    denom=0
    # print(labels)
    # print(output)
    for l, s in zip(labels, output):
        if l=="high risk":
            denom+=1
            if s=="high risk":
                num+=1



    # print(num, denom)
    high_risk_acc=float(num/denom)

    print("high risk accuracy: %0.2f"% (high_risk_acc))

    num=0
    denom=0
    for l, s in zip(labels, output):
        if l=="mid risk":
            denom+=1
            if s=="mid risk":
                num+=1

    print(num, denom)
    mid_risk_acc=float(num/denom)

    print("mid risk accuracy: %0.2f"% (mid_risk_acc))


    num=0
    denom=0
    for l, s in zip(labels, output):
        if l=="low risk":
            denom+=1
            if s=="low risk":
                num+=1

    print(num, denom)
    low_risk_acc=float(num/denom)

    print("low risk accuracy: %0.2f"% (low_risk_acc))


    # output = cross_validate(classifier, scaled, labels, cv=cv, scoring='accuracy', return_estimator=True)
    # for idx, estimator in enumerate(output['estimator']):
    #     feature_importances = pd.DataFrame(estimator.feature_importances_,
    #                                        index=features.columns,
    #                                        columns=['importance']).sort_values('importance', ascending=False)
    #     print(feature_importances)
    #
    #     feature_importances.plot(y='importance', kind='bar', use_index=True)
    #     plt.title("Feature importances using Mean Decrease in Impurity")
    #     plt.xlabel("Health factors")
    #     plt.ylabel("Mean decrease in impurity")
    #     plt.gcf().subplots_adjust(bottom=0.15)
    #     plt.show()

if __name__ == '__main__':
    main()
