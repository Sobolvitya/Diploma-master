from sklearn.datasets import load_breast_cancer
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def trainNN():
    data_set = load_breast_cancer()
    X = data_set['data']
    y = data_set['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(3, 3, 3))
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    return precision_score(y_test, predictions)

avg = 0
count = 500
for i in range(0, count):
    print("Step: " + str(i))
    avg += trainNN()
print("Final result is:")
print(avg/count)







