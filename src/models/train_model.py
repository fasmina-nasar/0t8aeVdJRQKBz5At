import pandas as pd

import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.filterwarnings("ignore")


def create_model(X_train,X_test,y_train,y_test,models,random_state=10):
  results=[]
  for model,params in models:
    start_time=time.time()
    model_instance=GridSearchCV(model(),params)
    model_instance.fit(X_train,y_train)
    elapsed_time=time.time()-start_time

    y_pred=model_instance.predict(X_test)
    f1=f1_score(y_test,y_pred)
    accuracy=accuracy_score(y_test,y_pred)
    roc_auc=roc_auc_score(y_test,y_pred)
    best_params=model_instance.best_params_
    results.append({'model': model.__name__, 'f1_score': f1, 'accuracy_score': accuracy, 'roc_auc_score': roc_auc, 'elapsed_time': elapsed_time, 'best_params': best_params})
  df=pd.DataFrame(results)
  return df
