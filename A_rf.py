from sklearn.ensemble import RandomForestClassifier
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
#print(data_train.shape)

#input testing set
test = read_csv(r"test_set.csv")
test = test.values
scaler= MinMaxScaler((0,1))
test=scaler.fit_transform(test)
y_test=test[:,1]
X_test=test[:,[5, 6, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19]]
#print(data_test.shape)

#start time
start=time.time()

#random forest
n_estimators_range=[100]#10,50,200,300
param_rf = dict(n_estimators=n_estimators_range)
clf_rf = GridSearchCV(RandomForestClassifier(),param_rf)
clf_rf.fit(X_train, y_train)
print(clf_rf.best_params_)
print(clf_rf.best_estimator_)
rf_pred = clf_rf.predict(X_train)
#accuracy_score of training set
#score_rf_train = accuracy_score(X_train, rf_pred)
#print('rf_train: ',score_rf_train)


print('rf----------')
#for testing data
rf_pred_test = clf_rf.predict(X_test)
#accuracy_score
score_rf = accuracy_score(y_test, rf_pred_test)
print('acc_test: ',score_rf)
#precison
pre_rf=average_precision_score(y_test, rf_pred_test)
print('pre_test: ',pre_rf)
#recall
rec_rf=recall_score(y_test, rf_pred_test)
print('recall_test: ',rec_rf)
#f1
f1_rf=f1_score(y_test, rf_pred_test)
print('f1_rf: ',f1_rf)
#auc
roc_rf=roc_auc_score(y_test, rf_pred_test)
print('roc_test: ',roc_rf)




#end time
end=time.time()
print('running time: %s seconds' %(end-start))
