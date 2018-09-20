import sys
import random
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn import tree
from sklearn.externals.six import StringIO
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
    # train_df['type'] = 'train'

    test_df = getDF(sys.argv[2])
    # train_df['type'] = 'test'

    # concat_df = pd.concat([train_df, test_df])
    #
    # dummies_df = pd.get_dummies(concat_df, columns=['date', 'opponent', 'location', 'in_top25', 'media', 'label'])
    #
    # train_df = dummies_df[dummies_df['type'] == 'train']
    # test_df = dummies_df[dummies_df['type'] == 'test']
    #
    # train_df = train_df.drop(['type', 'label_Lose'], axis=1)
    # test_df = test_df.drop(['type', 'label_Lose'], axis=1)


    # X_train = train_df[train_df[]]
    # Y_train = train_df[train_df['label_Win'] >= 0]

    # print(train_df)
    # input()
    # X_test = test_df.loc[:, 'location':'media']
    # Y_test = test_df.label

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

    classifier = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=8, min_samples_leaf=4)

    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    print (Y_test)
    print (Y_pred)

    print(classification_report(Y_test, Y_pred))

    makeGraph(classifier)
    return

if __name__=='__main__':
    main()
