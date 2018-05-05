from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import hidden_layers_size
from genetic_algo import get_zeroes_biases_vectors


def trainNN(nn_structure):
    data_set = load_digits()
    X = data_set['data']
    y = data_set['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers_size)
    mlp.fit(X_train, y_train)
    mlp.coefs_ = nn_structure
    mlp.intercepts_ = get_zeroes_biases_vectors()
    predictions = mlp.predict(X_test)
    return accuracy_score(y_test, predictions)

