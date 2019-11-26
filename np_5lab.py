import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#навчальні дані, що складаються з 4 прикладів - 3 вхідних значення та 1 вихід
training_inputs = np.array([[0,0],
                            [0,1],
                            [1,0],
                            [1,1]])

training_outputs = np.array([[0,1,1,1]]).T #xor - 0,1,1,0 #or - 0,1,1,1 #and - 0,0,0,1

np.random.seed(1)

synaptic_weights = 2 * np.random.rand(2,1) -1

print("Рандомні ваги:",synaptic_weights)

#Метод зворотнього поширення
for i in range(3000):
    input_layer = training_inputs
    outputs = sigmoid( np.dot( input_layer, synaptic_weights))

    err = training_outputs - outputs
    #корегування ваг
    adjustments = np.dot( input_layer.T, err * (outputs * (1 - outputs)))

    synaptic_weights += adjustments
    dj = -(training_outputs  * np.log(outputs)+(1 - training_outputs ) * np.log(1 -outputs))
print("Ваги після навчання: \n", synaptic_weights)
    
print("Результат після навчання: \n",outputs)

#test

new_inputs = np.array([[1,1],
                       [0,1]])

train = sigmoid(np.dot(new_inputs, synaptic_weights))
print("New:",train)

#Функція витрат
def plot_cost_function():
        plt.plot(dj, 'ro--')
        plt.xlabel("item")
        plt.ylabel("J")
        plt.title("COST FUNCTION")
        plt.legend(['J = -(1/m) * (y * np.log(a) + (1 - y) * np.log(1 - a))'])
        plt.grid(True)
        plt.show()
        
if __name__ == '__main__':
    a = plot_cost_function()
    print("Cost functions:",dj)
    
