from sklearn.ensemble import AdaBoostClassifier
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

X_test=test[:,[5, 6, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19]]

#start time
start=time.time()


#adaboosting  
n_estimators_range=[450]#250, 350, 550, 650
param_rf = dict(n_estimators=n_estimators_range)
clf_boost = GridSearchCV(AdaBoostClassifier(),param_rf)
clf_boost.fit(X_train, y_train)
print(clf_boost.best_estimator_)
boost_pred = clf_boost.predict(X_train)

print('Aba----------')
#for testing data
boost_pred = clf_boost.predict(X_test)
#accuracy_score
score_boost = accuracy_score(y_test, boost_pred)
print('acc_test: ',score_boost)
#precison
pre_boost=average_precision_score(y_test, boost_pred)
print('pre_test: ',pre_boost)
#recall
rec_boost=recall_score(y_test, boost_pred)
print('recall_test: ',rec_boost)
#f1
f1_boost=f1_score(y_test, boost_pred)
print('f1_boost: ',f1_boost)
#roc
roc_boost=roc_auc_score(y_test, boost_pred)
print('roc_test: ',roc_boost)

#end time
end=time.time()
print('running time: %s seconds' %(end-start))

