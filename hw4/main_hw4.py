import sys
import random
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.metrics import *
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as knn
import pydot


def getDF(csvFile):
    table = pd.read_csv(csvFile, sep= ',')
    return table


def makeGraph(classifier):
    dd = StringIO()
    tree.export_graphviz(classifier, out_file=dd)
    graph = pydot.graph_from_dot_data(dd.getvalue())
    graph[0].write_pdf('hw2' + str(random.randrange(1000)) + '.pdf')

    return


def main():
    df = getDF(sys.argv[1])
    df_shuffled = df.sample(frac=1)
    df = df_shuffled.reset_index(drop=True)

    df_list = []
    for i in range(0, 150, 30):
        df_list.append(df[i:i+30])

    Columns = ['sepal_l','sepal_w','petal_l','petal_w','result']
    train_knn = pd.DataFrame(columns=Columns)
    test_knn = pd.DataFrame(columns=Columns)
    k_vals = {}
    all_accuracies = []
    for i in range(5):
        print(i)
        test_knn = pd.concat([test_knn, df_list[i]])

        for j in range(5):
            if i != j:
                train_knn = pd.concat([train_knn, df_list[j]])
        
        x_test_knn = test_knn.loc[:, :'petal_w']
        y_test_knn = test_knn.loc[:, ['result']]

        x_train_knn = train_knn.loc[:, :'petal_w']
        y_train_knn = train_knn.loc[:, ['result']]

        accuracies = []
        for i in range(1, 51):
            knn_classifier = knn(n_neighbors=i)
            knn_classifier.fit(x_train_knn, y_train_knn)
            knn_y_pred = knn_classifier.predict(x_test_knn)
            # print(accuracy_score(y_test_knn, knn_y_pred))
            accuracies.append(accuracy_score(y_test_knn, knn_y_pred))
        
        all_accuracies.append(accuracies)
        tot = 0

        for i in range(50):
            for l in all_accuracies:
                tot += l[i]
            tot = tot / 5
            k_vals[i+1] = tot
            tot = 0
        
        print (k_vals)
        test_knn = pd.DataFrame(columns=Columns)
        train_knn = pd.DataFrame(columns=Columns)

    print (k_vals)

    num = 1
    for training_df in df_list[1:]:
        x_train = training_df.loc[:, :'petal_w']
        y_train = training_df.loc[:, ['result']]
        
        knn_classifier = knn(n_neighbors=2)
        knn_classifier.fit(x_train, y_train)

        dtree_classifier = dtc(criterion='entropy', random_state=100, max_depth=8, min_samples_leaf=4)
        dtree_classifier.fit(x_train, y_train)

        knn_y_pred = knn_classifier.predict(x_test)
        dtree_y_pred = dtree_classifier.predict(x_test)

        # print('{}: Dtree Accuracy - {}'.format(num, accuracy_score(y_test, dtree_y_pred)))
        # print('Report: {}'.format(classification_report(y_test, dtree_y_pred)))

        # print('{}: KNN Accuracy - {}'.format(num, accuracy_score(y_test, knn_y_pred)))
        # print('Report: {}'.format(classification_report(y_test, knn_y_pred)))

        # print('-'*50)
        num += 1
    return

if __name__=='__main__':
    main()
