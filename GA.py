from brain import *
from pprint import pprint

population = 10

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0, 1, 1, 0]]
X = np.asarray(X)
X = X.T
Y = np.asarray(Y)

print(X)
print(Y)

def main():

    # Test code for verifying score
    # node = brain(2, 4, 1)
    # node.show_weights()
    # for i in range(30):
    #     node.train(X, Y)
    #     node.score = score_eval(node, X, Y)
    #
    #     print(node.score)
    # print(node.predict(X)[1])


    brains = create_swarm(9, 9, 9)
    highest = brains[0].score
    for i in range(10):
        new_brains = []
        calculateFitness(brains)

        new_brains = newGeneration(brains)


    print("Finished")


#==============================================================================
# Helper functions
def create_swarm(n_x, n_h, n_y):
    brains = []
    for i in range(population):
        brains.append(brain(n_x, n_h, n_y))


    brains = np.asarray(brains)
    return brains


def newGeneration(networks):
    #Sorts by fitness scores
    networks = sorted(networks, key=lambda x: x.fitness, reverse = True)
    #Do something to pick based on fitness
    children = []
    for network in networks:
        children.append(pickOne(networks))



    children = np.asarray(children)
    return children


def pickOne(networks):
    index = 0

    r = np.random.rand()
    while(r > 0):
        r = r - networks[index].fitness
        index = index + 1
    index = index - 1

    child = networks[index]

    child.mutate(0.1)


    return child


def calculateFitness(networks):
    sum = 0


    for network in networks:

        network.score = score_eval(network, X, Y)

    for network in networks:
        sum = sum + network.score
    #print(sum)
    for network in networks:
        network.assign_fitness(network.score/sum)


def score_eval(network, X, Y):
    if(isinstance(network, brain)):
        m = Y.shape[1]

        A1, A2 = network.predict(X)
        print(A2)
        cost = int(input("How well was it?:"))
        # cost = -np.sum(np.multiply(Y, np.log(A2)) + np.multiply(1 - Y, np.log(1 - A2)))
        # print(cost)
    return (cost)

if __name__ == "__main__":
    main()
