import numpy as np
from activation_function import BinaryStep

class Perceptron:
    def __init__(self, input_values, output_values, learning_rate=1e-2, activation_function=BinaryStep):
        self.input_values = input_values
        self.output_values = output_values
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.W = np.random.rand(len(input_values[0]))
        self.theta = np.random.rand(1)[0]
        self.epochs = 0
        
    def train(self):

        error = True
        while error:
            self.epochs += 1
            print(f'Épocas {self.epochs}')
            error = False
            for x, d in zip(self.input_values, self.output_values):
                u = np.dot(np.transpose(x), self.W) - self.theta
                y = self.activation_function.g(u)
                                
                print(f'O valor encontrado {y} e o esperado é de {d}')
                if y != d:
                    print(f'Ajustando pesos')
                    print(f'Valores pré ajuste')
                    print(f'\t W: {self.W}')
                    print(f'\t theta: {self.theta}')
                    self.theta = self.theta + self.learning_rate  * (d - y) * -1
                    self.W = self.W + self.learning_rate * (d - y) * x
                    error = True
                    print(f'Valores pós ajuste')
                    print(f'\t W: {self.W}')
                    print(f'\t theta: {self.theta}')
                    break
            if not error:
                print(f'Processo de treinamento concluído com sucesso')
                print(f'\t W: {self.W}')
                print(f'\t theta: {self.theta}')
                
            print('')
            
    def testes(self):
        
        while self.epochs <= 5:
            self.epochs += 1
            print('Epoca {}'.format(self.epochs))
            for x, d in zip(self.input_values, self.output_values):
                u = np.dot(np.transpose(x), self.W) - self.theta
                y = self.activation_function.g(u)
                if y >= 0:
                    y = 1
                else:
                    y = -1
                print('O valor encontrado foi {}'.format(y))
            
                    
    def evaluate(self, input_values):
        u = np.dot(np.transpose(input_values), self.W) - self.theta
        return self.activation_function.g(u)
             
               
    