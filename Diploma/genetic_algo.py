import numpy as np
import operator
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import hidden_layers_size

input_layer_size = 30
output_layer_size = 1

size_of_population = 20

def trainNN(nn_structure):
    data_set = load_breast_cancer()
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

def generate_basic_structure_with_zeroes():
    NN = []
    all_layers_size = (input_layer_size,) + hidden_layers_size + (output_layer_size,)
    for i in range(0, len(all_layers_size) - 1):
        NN.append(np.zeros((all_layers_size[i], all_layers_size[i + 1])))

    print("Successfully generated basic NN topology")
    return NN

def get_zeroes_biases_vectors():
    biases = []
    all_layers_size = (input_layer_size,) + hidden_layers_size + (output_layer_size,)
    for i in range(0, len(all_layers_size) - 1):
        biases.append(np.zeros(all_layers_size[i + 1]))

    print("Successfully generated basic biases vector")
    return biases

def run_genetic_algo():
    population = [generate_basic_structure_with_zeroes() for _ in range(0, size_of_population)]
    for _ in range(1, 2):
        # population = crossover(mutate(get_the_best_half(population)))
        population = get_the_best_half(population)
        # print(population)
    return get_the_best_one(population)

def get_the_best_half(population):
    topology_fitness_map = {}
    for i in range(0, len(population)):
        topology_fitness_map[i] = trainNN(population[i])
    best_topologies = sorted(topology_fitness_map.items(), key=operator.itemgetter(1), reverse=True)[:int(len(topology_fitness_map)/2)]
    return list(map(lambda topology: topology[1], best_topologies))

def mutate(population):
    result = []
    return result

def crossover(population):
    result = []
    return result

def get_the_best_one(population):
    return population[0]

run_genetic_algo()

