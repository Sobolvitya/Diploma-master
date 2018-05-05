import numpy as np
import operator
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from random import randint

from utils import get_zeroes_biases_vectors

'''
This code tries to generate weight for NN and beat the back propagation algo
'''
from config import hidden_layers_size

input_layer_size = 30
output_layer_size = 1
max_iterations = 20
size_of_population = 12

mutation_factor = 10
acceptable_value = 0.9

def trainNNWithStructure(mlp, nn_structure, X_test, y_test):
    mlp.coefs_ = nn_structure
    predictions = mlp.predict(X_test)
    return accuracy_score(y_test, predictions)


def generate_NN():
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
    mlp.intercepts_ = get_zeroes_biases_vectors(input_layer_size, hidden_layers_size, output_layer_size)
    return X_test, mlp, y_test


def generate_basic_structure_with_random_values():
    NN = []
    all_layers_size = (input_layer_size,) + hidden_layers_size + (output_layer_size,)
    for i in range(0, len(all_layers_size) - 1):
        NN.append(np.random.random((all_layers_size[i], all_layers_size[i + 1])))
    return NN

def run_genetic_algo():
    X_test, mlp, y_test = generate_NN()
    population = [generate_basic_structure_with_random_values() for _ in range(0, size_of_population)]
    max_accuracy = 0
    for _ in range(max_iterations):
        half = get_the_best_half(mlp, population, X_test, y_test)
        # print("The best topology")
        # print(half[0])
        the_best_result = trainNNWithStructure(mlp, half[0], X_test, y_test)
        if (the_best_result > max_accuracy) :
            max_accuracy = the_best_result
        print("Current best result:  " + str(the_best_result))
        if (the_best_result > acceptable_value) :
            print("This should be enough. \n NN structure is: ")
            print(half[0])
            break
        population = crossover(mutate(half, the_best_result))
    last = trainNNWithStructure(mlp, get_the_best_one(get_the_best_half(mlp, population, X_test, y_test)), X_test, y_test)
    return max(last, max_accuracy)

def get_the_best_half(mlp, population, X_test, y_test):
    topology_fitness_map = {}
    for i in range(0, len(population)):
        topology_fitness_map[i] = trainNNWithStructure(mlp, population[i], X_test, y_test)
    best_topologies = sorted(topology_fitness_map.items(), key=operator.itemgetter(1), reverse=True)[:int(len(topology_fitness_map)/4)]
    # print(list(map(lambda topology: topology[1], best_topologies)))
    return list(map(lambda topology: population[topology[0]], best_topologies))

def mutate(populations, mutation_coef = None):
    adjusted_mutation_factor = int (mutation_factor / mutation_coef)
    for population in populations:
        for _ in range(adjusted_mutation_factor):
            layer = randint(0, len(population)-1)
            weigh = randint(0, len(population[layer])-1)
            population[layer][weigh] = (np.random.random(1)[0]/(10 ** np.random.randint(0, 2))) * ((-1) ** randint(1, 2))
    return populations

def crossover(population): #should be improved
    result = []
    for _ in range(size_of_population):
        tempNN1 = population[np.random.randint(0, len(population))]
        tempNN2 = population[np.random.randint(0, len(population))]
        tmpNNs = [tempNN1, tempNN2]
        tempResult = []
        for i in range(len(tempNN1)):
            tmpLayer = []
            for j in range(len(tempNN2[i])):
                tmpLayer.append(tmpNNs[np.random.randint(0, 2)][i][j])
            tempResult.append(np.array(tmpLayer))
        result.append(tempResult)
    return result

def get_the_best_one(population):
    return population[0]

best_one = run_genetic_algo()
print("Final result: " + str(best_one))

