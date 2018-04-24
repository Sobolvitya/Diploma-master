from sklearn.datasets import load_breast_cancer
from sklearn.metrics import precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import hidden_layers_size
from genetic_algo import generate_basic_structure_with_zeroes, get_zeroes_biases_vectors


def trainNN():
    data_set = load_breast_cancer()
    X = data_set['data']
    y = data_set['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers_size)
    # mlp.fit(X_train, y_train)
    mlp.coefs_ = generate_basic_structure_with_zeroes()
    mlp.intercepts_ = get_zeroes_biases_vectors()
    mlp.n_outputs_ = 1
    mlp.n_layers_ = 5
    mlp.out_activation_ = 'logistic'
    print(mlp._label_binarizer)
    predictions = mlp.predict(X_test)
    return accuracy_score(y_test, predictions)

avg = 0
count = 20
for i in range(0, count):
    print("Step: " + str(i))
    avg += trainNN()
print("Final result is:")
print(avg/count)







