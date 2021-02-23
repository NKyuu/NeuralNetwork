import numpy as np


class Layer:

    def __init__(self, size, link):
        self.size = link
        self.next = next
        self.w = np.random.randn(size, link)
        self.b = np.zeros((1, link))
        self.input = []
        self.output = []
        self.delta = []


class Network:
    # inputs: n of inputs
    # hidden: an array where length is the number of layers and the value the number of neurons
    # outputs: n of outputs

    dna = {}

    def __init__(self, inputs, hidden, outputs):
        if len(hidden) >= 1:
            self.dna['l1'] = Layer(inputs, hidden[0])

            if len(hidden) == 1:
                self.dna['l2'] = Layer(hidden[0], outputs)
                self.dna['size'] = 2

            elif len(hidden) == 2:
                self.dna['l2'] = np.random.randn(hidden[0], hidden[1])
                self.dna['l3'] = np.random.randn(hidden[1], outputs)
                self.dna['size'] = 3

            else:
                for i in range(len(hidden)):
                    if i == hidden[len(hidden) - 1]:
                        self.dna[self.__get_layer(1 + i)] = np.random.randn(hidden[i], outputs)
                    else:
                        self.dna[self.__get_layer(1 + i)] = np.random.randn(hidden[i], hidden[i + 1])

                self.dna[self.__get_layer(len(hidden) + 1)] = np.random.randn(hidden[len(hidden) - 1], outputs)
                self.dna['size'] = len(hidden) + 1
        else:
            self.dna['w1'] = np.random.randn(inputs, outputs)
            self.dna['b1'] = np.zeros((1, outputs))
            self.dna['size'] = 1

    def __get_layer(self, i):
        return "l" + str(i)

    def size(self):
        return self.dna.get("size")

    def printDNA(self):
        print(self.dna)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoider(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def __feedfoward(self, dataset):
        size = self.size()

        output = dataset
        no_sigs = [dataset]
        sigs = [0]

        for i in range(1, size + 1):
            self.dna[self.__get_layer(i)].input = output
            output = np.dot(output, self.dna[self.__get_layer(i)].w) + self.dna[self.__get_layer(i)].b
            no_sigs.append(output)
            output = self.sigmoid(output)
            self.dna[self.__get_layer(i)].output = output
            sigs.append(output)

        return output

    def predict(self, inputs):
        output = self.__feedfoward(inputs)
        return output

    def mse(self, real, predict):

        x = (np.array(predict) - np.array(real)) ** 2
        x = np.mean(x)

        y = np.array(predict) - np.array(real)

        return x, y

    def train(self, epoch, dataset, answers, lr):
        for i in range(epoch):
            output = self.__feedfoward(dataset)
            #deltas error
