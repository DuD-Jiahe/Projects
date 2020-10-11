from sklearn import tree
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import time

#input training set
train = read_csv(r"new_data.csv")
train = train.values
scaler= MinMaxScaler((0,1))
train=scaler.fit_transform(train)
y_train=train[:,1]
X_train=train[:,[5, 6, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19]]

#input testing set
test = read_csv(r"test_set.csv")
test = test.values
scaler= MinMaxScaler((0,1))
test=scaler.fit_transform(test)
y_test=test[:,1]
X_test=test[:,[5, 6, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19]]

X_test=test[:,[5, 6, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19]]#4:19]
#print(data_test.shape)

#start time
start=time.time()


#Decision Tree
max_depth_range=[8]#range(1,10)
param_dt = dict(max_depth=max_depth_range)
clf_dt = GridSearchCV(tree.DecisionTreeClassifier(), param_dt)
clf_dt.fit(X_train, y_train)
print(clf_dt.best_params_)
print(clf_dt.best_estimator_)
dt_pred_train = clf_dt.predict(X_train)



print('dt----------')
#for testing data
dt_pred_test = clf_dt.predict(X_test)
#accuracy_score
score_dt = accuracy_score(y_test, dt_pred_test)
print('acc_test: ',score_dt)
#precison
pre_dt=average_precision_score(y_test, dt_pred_test)
print('pre_test: ',pre_dt)
#recall
rec_dt=recall_score(y_test, dt_pred_test)
print('recall_test: ',rec_dt)
#f1
f1_dt=f1_score(y_test, dt_pred_test)
print('f1_test: ',f1_dt)
#roc
auc_dt=roc_auc_score(y_test, dt_pred_test)
print('roc_test: ',auc_dt)


#end time
end=time.time()
print('running time: %s seconds' %(end-start))