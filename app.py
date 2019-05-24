import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

graduates_data = pd.read_csv('score_board_train.csv', delimiter=";")

le = preprocessing.LabelEncoder()

#convert the categorical columns into numeric
graduates_data['accepted'] = le.fit_transform(graduates_data['accepted'])
graduates_data['graduated'] = le.fit_transform(graduates_data['graduated'])
graduates_data['id'] = le.fit_transform(graduates_data['id'])

graduates_data['art_exam'] = graduates_data['art_exam']*100
graduates_data['art_exam'] = graduates_data['art_exam'].round(decimals = 3)

graduates_data['maths_exam'] = graduates_data['maths_exam']*100
graduates_data['maths_exam'] = graduates_data['maths_exam'].round(decimals = 3)

graduates_data['language_exam'] = graduates_data['language_exam']*100
graduates_data['language_exam'] = graduates_data['language_exam'].round(decimals = 3)

graduates_data['interview_score'] = graduates_data['interview_score']*100
graduates_data['interview_score'] = graduates_data['interview_score'].round(decimals = 3)

graduates_data['essay_score'] = graduates_data['essay_score']*100
graduates_data['essay_score'] = graduates_data['essay_score'].round(decimals = 2)

graduates_data['gpa'] = graduates_data['gpa'].round(decimals = 3)

cols = [col for col in graduates_data.columns \
        if col not in ['id','graduated', 'accepted', 'year']]

data = graduates_data[cols]
target = graduates_data['graduated']

#split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.15, random_state = 10)

gauss_model = GaussianNB()
gauss_model.fit(data_train, target_train)
y_pred = gauss_model.predict(data_test)
print ("Gaussian score : ",accuracy_score(target_test, y_pred))

neigh_model = KNeighborsClassifier(n_neighbors=50, algorithm='auto')
neigh_model.fit(data_train, target_train)
pred = neigh_model.predict(data_test)
print ("Kneighbors accuracy score : ",accuracy_score(target_test, pred))

svc_model = LinearSVC(random_state=0,max_iter = 5000)
pred = svc_model.fit(data_train, target_train).predict(data_test)
print("LinearSVC accuracy : ",accuracy_score(target_test, pred, normalize = True))

rfc_model = RandomForestClassifier(n_estimators=100)
pred_rfc = rfc_model.fit(data_train, target_train).predict(data_test)
print("RandomForestClassifier : ",accuracy_score(target_test, pred_rfc, normalize = True))

mlp_model = MLPClassifier()
pred_mlp = mlp_model.fit(data_train, target_train).predict(data_test)
print("MLP accuracy : ",accuracy_score(target_test, pred_mlp, normalize = True))

#ada_model = AdaBoostClassifier()
#pred_ada = ada_model.fit(data_train,target_train).predict(data_test)
#print("AdaBoost accuracy : ",accuracy_score(target_test, pred_ada, normalize = True))

pickle.dump(neigh_model, open("model_neigh.pkl","wb"))