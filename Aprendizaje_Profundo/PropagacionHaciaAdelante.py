# Objetivos:
# construir una red neuronal
# Calcular la suma ponderada en cada nodo
# Calcular la activación del nodo
# Usar la propagación hacia adelante para propagar datos

import numpy as np # import Numpy library to generate 

weights = np.around(np.random.uniform(size=6), decimals=2) # initialize the weights
biases = np.around(np.random.uniform(size=3), decimals=2) # initialize the biases

#peso
print(weights)
#base
print(biases)

#entradas
x_1 = 0.5 # input 1
x_2 = 0.85 # input 2

print('x1 is {} and x2 is {}'.format(x_1, x_2))

#Suma ponderada:
z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]

print('La suma ponderada de las entradas en el primer nodo de la capa oculta es {}'.format(z_11))

# Suma ponderada segundo nodo
z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]

print('La suma ponderada de las entradas en el segundo nodo de la capa oculta es {}'.format(np.around(z_12, decimals=4)))

# Suponiendo una función de activación sigmoidea

a_11 = 1.0 / (1.0 + np.exp(-z_11))

print('La activación del primer nodo de la capa oculta es {}'.format(np.around(a_11, decimals=4)))

a_12 = 1.0 / (1.0 + np.exp(-z_12))

print('La activación del segundo nodo de la capa oculta es {}'.format(np.around(a_12, decimals=4)))

# Ahora, estas activaciones servirán como entradas para la capa de salida. Calculemos la suma ponderada de estas entradas para el nodo en la capa de salida. Asigna el valor a z_2.

z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]

print('La suma ponderada de las entradas en el nodo en la capa de salida es {}'.format(np.around(z_2, decimals=4)))

#funcion sigmoidea para nuestra nodo de salida
a_2 = 1.0 / (1.0 + np.exp(-z_2))


print('La salida de la red para x1 = 0,5 y x2 = 0,85 es {}'.format(np.around(a_2, decimals=4)))



#Ahora realizaremos un modelo mas autoamtizado, ya que anteriormente estabamos realizando cada operacion de manera manual, lo cual no es eficiente

# Comencemos por definir formalmente la estructura de la red.

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):

    num_nodes_previous = num_inputs # numero de nodos por cada capa

    network = {} # inicializamos la red en un diccionario

    for layer in range(num_hidden_layers + 1): 
        
        # determinamos los nombres de las capas
        if layer == num_hidden_layers:
            layer_name = 'output'
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1)
            num_nodes = num_nodes_hidden[layer]
        
        # inicializamos el peso y la base por cada capa 
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
        
        num_nodes_previous = num_nodes
    
    return network # retornamos el diccionario


#inicializamos la funcion con 5 entradas, 3 capas ocultas, la primera copa con 3 nodos, la sgunda capa con 2 nodos, la tercera capa con 3 nodos, y la capa de la salida con 1 nodo
small_network = initialize_network(5, 3, [3, 2, 3], 1)

print ("hola-----------------------", small_network)

#ahora calcularemos la suma ponderada de cada nodo 

def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

from random import seed

np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2)

print('Las entradas de la red son {}'.format(inputs))



node_weights = small_network['layer_1']['node_1']['weights']
node_bias = small_network['layer_1']['node_1']['bias']

weighted_sum = compute_weighted_sum(inputs, node_weights, node_bias)
print('La suma ponderada del primer nodo de la primera capa oculta es  {}'.format(np.around(weighted_sum[0], decimals=4)))