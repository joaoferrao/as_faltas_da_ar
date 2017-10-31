from random import randint
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np

X_predict = np.array([[0, 1, 0, 0, 0, 1, 0]])
randomstate = randint(0, 100)
c_values = [0.01, 0.1, 1, 10]
kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']
activ_nn = ['logistic', 'relu', 'identity']
solver_nn = ['lbfgs', 'sgd', 'adam']
raw_data = pd.read_csv('presencas_ar.csv', sep=',')
lista_ids = pd.read_csv('lista_ids.csv').values

def create_dep_dataset(deputado_id):
    # Experimentar filtrar um deputado_id.
    deputado_id_filter = deputado_id
    pd_train_all_dep = raw_data.drop(['data', 'dia_semana'], axis=1)
    # Filtrar apenas o deputado que nos interesa
    pd_train = pd_train_all_dep.loc[pd_train_all_dep['deputado_id'] == deputado_id_filter]
    # Agora já não nos interessa a coluna "deputado id"
    pd_train = pd_train.drop(['deputado_id'], axis=1)
    # Agora temos que transformar o DataFrame em Numpy, aproveitar e separar em test,train
    # Criar Y_data
    y_train_temp = pd_train.filter(items=["presente"]).values
    y_train = y_train_temp.flatten()
    # Criar X_data
    X_train = pd_train.filter(items=[
                                "segunda_feira",
                                "terca_feira",
                                "quarta_feira",
                                "quinta_feira",
                                "sexta_feira",
                                "antes_de_feriado",
                                "depois_de_feriado"
                            ]).values

    return X_train, y_train


def train_test_dep_data(deputado_id):
    X_train, y_train = create_dep_dataset(deputado_id)
    # now we use the model_selection to split our data in trainning and cross validation
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_train,
                                                                                y_train,
                                                                                test_size=0.2,
                                                                                random_state=randomstate)
    # start trying with linear regression
    # start a loop to itirate over all the parameters we are trying out
    final_score = 0
    prediction = 0
    for c in c_values:
        for kern in kernel_types:
            SVM_clf = SVC(C=c, kernel=kern, gamma='auto', random_state=randomstate)
            SVM_clf.fit(X_train_final, y_train_final)
            score = SVM_clf.score(X_test_final, y_test_final)
            if score >= final_score:
                final_score = score
                prediction = SVM_clf.predict(X_predict)


            # print("Support Vector Machines | C = %f | Kernel = %s | Score = %f" % (
            #     c, kern, score))

    # for c in c_values:
    #     LR_clf = LogisticRegression(C=c, random_state=randomstate)
    #     LR_clf.fit(X_train_final, y_train_final)
    #     score = LR_clf.score(X_test_final, y_test_final)
    #     print("Linear Regression | C = %f | Score = %f" % (c, score))

    # continue our models test with SVMs
    # now we try with sklearn neural network
    # for activ in activ_nn:
    #     for solv in solver_nn:
    #         nn_clf = MLPClassifier(hidden_layer_sizes=(500, 400), activation=activ,
    #                                solver=solv,
    #                                max_iter=1000)
    #         nn_clf.fit(X_train_final, y_train_final)
    #         score = nn_clf.score(X_test_final, y_test_final)
    #         print("Neural Networks | Activation = %s | Solver = %s | Score = %f" % (
    #             activ, solv, score))
    results_dic = {'id_deputado': deputado_id, 'pontuacao': score, 'previsao':
        prediction[0]}
    return results_dic


pd_final_pred = pd.DataFrame(columns=['id_deputado', 'pontuacao', 'previsao'])

for dep in np.nditer(lista_ids):
    print(train_test_dep_data(deputado_id=dep))


    # pd_final_pred.loc[dep]['id_deputado'] = temp_dic['id_deputado']


