from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def split_data(X,y,test_size,random_state):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state)
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    print(X_train.shape,X_test.shape)
    print(y_train)
    print(y_test)
    return X_train,X_test,y_train,y_test