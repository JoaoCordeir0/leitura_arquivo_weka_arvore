import pandas as pd
from scipy.io.arff import loadarff

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

def readFile(file_path: str):       
    data = loadarff(file_path)
    df = pd.DataFrame(data[0], columns=['Tempo', 'Temperatura', 'Umidade', 'Vento', 'Jogar'])
    return df

def createTree(file_path: str):
    # Get table datas
    table_data = readFile(file_path)

    # Divide collumns between the X - attributes and y - result 
    X = table_data[['Tempo', 'Temperatura', 'Umidade', 'Vento']]
    y = table_data['Jogar']

    # Hot encode to transform all string values to int, so that the decision tree can read 
    # them and analyze
    X = pd.get_dummies(X, columns=['Tempo', 'Temperatura', 'Umidade', 'Vento'], drop_first=True)
        
    # Instance DecisionTree class ang give wich criterion will use
    clf = DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(X, y.astype('str'))

    # Generate tree figure
    plt.figure(figsize=(12,8))
    tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Nao', 'Sim'], rounded=True)
    plt.show()

createTree('./model/jogar_tenis.arff')
