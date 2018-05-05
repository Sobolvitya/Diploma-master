from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from utils import get_zeroes_biases_vectors

input_layer_size = 30
output_layer_size = 1
hidden_layer_sizes=(3,)
def trainNN():
    data_set = load_breast_cancer()
    X = data_set['data']
    print(data_set['data'].shape)
    y = data_set['target']
    print(data_set['target'].shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes)
    mlp.fit(X_train, y_train)
    mlp.intercepts_ = get_zeroes_biases_vectors(input_layer_size, hidden_layer_sizes, output_layer_size)
    predictions = mlp.predict(X_test)
    print(mlp.coefs_)
    # print(confusion_matrix(y_test,predictions))
    # print(classification_report(y_test,predictions))
    return accuracy_score(y_test, predictions)

avg = 0
count = 1
print("NN results")
for i in range(0, count):
    print("Step: " + str(i))
    avg += trainNN()
print("Final result is:")
print(avg/count)







