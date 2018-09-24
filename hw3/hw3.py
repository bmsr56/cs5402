import sys
import random
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.naive_bayes import GaussianNB
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


    train_df = getDF(sys.argv[1])
    test_df = getDF(sys.argv[2])


    X_train = pd.get_dummies(train_df.loc[:, 'location':'media'])
    Y_train = pd.get_dummies(train_df.label)
    Y_train = Y_train.drop(['Lose'], axis=1)

    X_test =  pd.get_dummies(test_df.loc[:, 'location':'media'])
    Y_test = pd.get_dummies(test_df.label)
    Y_test = Y_test.drop(['Lose'], axis=1)

    print(X_train)
    print('-'*50)
    print(Y_train)
    print('-'*50)
    print(X_test)
    print('-'*50)
    print(Y_test)

    classifier = GaussianNB()

    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    print (Y_test)
    print (Y_pred)

    print(classification_report(Y_test, Y_pred))

    makeGraph(classifier)
    return

if __name__=='__main__':
    main()
