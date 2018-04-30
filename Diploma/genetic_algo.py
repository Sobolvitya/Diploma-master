import numpy as np
import operator
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from random import randint


from config import hidden_layers_size

input_layer_size = 30
output_layer_size = 1

size_of_population = 20

mutation_factor = 10

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

    # print("Successfully generated basic NN topology")
    return NN

def get_zeroes_biases_vectors():
    biases = []
    all_layers_size = (input_layer_size,) + hidden_layers_size + (output_layer_size,)
    for i in range(0, len(all_layers_size) - 1):
        biases.append(np.zeros(all_layers_size[i + 1]))

    # print("Successfully generated basic biases vector")
    return biases

def run_genetic_algo():
    population = [generate_basic_structure_with_zeroes() for _ in range(0, size_of_population)]
    for _ in range(1, 20):
        # population = crossover(mutate(get_the_best_half(population)))
        half = get_the_best_half(population)
        print("Current best result:  " + str(trainNN(half[0])))
        population = crossover(mutate(half))

        # print(population)
    return get_the_best_one(get_the_best_half(population))

def get_the_best_half(population):
    topology_fitness_map = {}
    for i in range(0, len(population)):
        topology_fitness_map[i] = trainNN(population[i])
    best_topologies = sorted(topology_fitness_map.items(), key=operator.itemgetter(1), reverse=True)[:3 * int(len(topology_fitness_map)/4)]
    # print(list(map(lambda topology: topology[1], best_topologies)))
    return list(map(lambda topology: population[topology[0]], best_topologies))

def mutate(populations):
    for population in populations:
        for _ in range(mutation_factor):
            layer = randint(0, len(population)-1)
            weigh = randint(0, len(population[layer])-1)
            population[layer][weigh] = np.random.random(1)[0]/10
    return populations

def crossover(population):
    result = []
    for _ in range(size_of_population):
        tempNN1 = population[np.random.randint(0, len(population))]
        tempNN2 = population[np.random.randint(0, len(population))]
        tmpNNs = [tempNN1, tempNN2]
        tempResult = []
        for i in range(len(tempNN1)):
            tempResult.append(tmpNNs[np.random.randint(0, 2)][i])
        result.append(tempResult)
    return result

def get_the_best_one(population):
    return population[0]

best_one = run_genetic_algo()
print("Final result: " + str(trainNN(best_one)))

