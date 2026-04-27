import matplotlib.pyplot as plt
import numpy as np
import random as rd



def generate_x(): # Generate a random input vector and conjunctions
    x1 = rd.randint(0,1)
    x2 = rd.randint(0,1)
    x3 = rd.randint(0,1)
    x4 = rd.randint(0,1)
    x5  = x1 * x2
    x6  = x1 * x3
    x7  = x1 * x4
    x8  = x2 * x3
    x9  = x2 * x4
    x10 = x3 * x4
    x11 = x1 * x2 * x3
    x12 = x1 * x2 * x4
    x13 = x1 * x3 * x4
    x14 = x2 * x3 * x4
    x15 = x1 * x2 * x3 * x4

    x = np.array([1, x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15])
    return x

def generate_x(pair, val1, val2): # Generate an input vector with two specified values
    x = np.zeros(16)
    x[0] = 1
    x[pair[0]] = val1
    x[pair[1]] = val2
    x[5]  = x[1] * x[2]
    x[6]  = x[1] * x[3]
    x[7]  = x[1] * x[4]
    x[8]  = x[2] * x[3]
    x[9]  = x[2] * x[4]
    x[10] = x[3] * x[4]
    x[11] = x[1] * x[2] * x[3]
    x[12] = x[1] * x[2] * x[4]
    x[13] = x[1] * x[3] * x[4]
    x[14] = x[2] * x[3] * x[4]
    x[15] = x[1] * x[2] * x[3] * x[4]

    
    return x

def initialize_w(): # Generate a Random weight vector
    w = np.zeros(16, dtype=float)

    for i in range(0, len(w)):
        w[i] = rd.random()
    return w

def perceptron(x, w): # Mcculloch Perceptron 
    sum = np.dot(x, w)
    if(sum >= 0):
        return 1
    return 0

def generate_data(N): # Generates a set of random input vectors
    data = []
    for i in range(N):
        data.append(generate_x())
    return np.array(data)

def uneven_bits(x): # Target function
    if np.sum(x[1:5]) %2 == 1:
        return 1
    return 0

def train(w): # Generates all possible input vectors and optimizes w 
    errors = 0
    epoch = 1
    errorshistory = []
    
    done = False
    while not done:
        done = True
        errors = 0
        for x1 in range(0,2):
            for x2 in range(0,2):
                for x3 in range(0,2):
                    for x4 in range(0,2):
                        x5  = x1 * x2
                        x6  = x1 * x3
                        x7  = x1 * x4
                        x8  = x2 * x3
                        x9  = x2 * x4
                        x10 = x3 * x4
                        x11 = x1 * x2 * x3
                        x12 = x1 * x2 * x4
                        x13 = x1 * x3 * x4
                        x14 = x2 * x3 * x4
                        x15 = x1 * x2 * x3 * x4
                        x = np.array([1, x1, x2, x3, x4, x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15])
                        target = uneven_bits(x)
                       
                        res = perceptron(x, w)
                        if res != target:
                            errors = errors + 1
                            w = w + (target - res) * x  # standard delta rule
                            done = False
        epoch = epoch + 1
        errorshistory.append([epoch, errors])               
    return w, errorshistory


w = initialize_w()
#w = np.zeros(len(w))
w, errorhist = train(w)
errorhist = np.array(errorhist)

print("Gewichte w = ", w)

plt.plot(errorhist[:,0], errorhist[:, 1])
plt.grid()
plt.ylabel("Anzahl Fehler")
plt.xlabel("Anzahl Trainingsläufe")
plt.show()


pairs = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
steps = np.arange(0, 1.1, 0.1)

### Plotting logic

for pair in pairs:
    fig, ax = plt.subplots()
    plot = []
    for val1 in steps:
        row = []
        for val2 in steps:
            x = generate_x(pair, val1, val2)
            row.append(perceptron(x, w))
        plot.append(row)
    plt.imshow(np.flipud(plot), interpolation="nearest")
    
    ax.set_xticks(10*steps)
    ax.set_yticks(10*np.flip(steps))
    ax.set_xticklabels([f"{s:.1f}" for s in steps], rotation=45, fontsize=7)
    ax.set_yticklabels([f"{s:.1f}" for s in steps], fontsize=7)
    ax.set_title(f"Perceptron Output $ \quad (x_{pair[0]}, x_{pair[1]})$")
    ax.set_ylabel(f"$x_{pair[0]}$")
    ax.set_xlabel(f"$x_{pair[1]}$")
    plt.savefig("Perceptron" +  str(pair), dpi=400)
    plt.show()

for pair in pairs:
    
    fig, ax = plt.subplots()
    plot = []
    for val1 in steps:
        row = []
        for val2 in steps:
            x = generate_x(pair, val1, val2)
            row.append(np.dot(x, w))
        plot.append(row)
    plt.imshow(np.flipud(plot), interpolation="nearest")
    
    ax.set_xticks(10*steps)
    ax.set_yticks(10*np.flip(steps))
    ax.set_xticklabels([f"{s:.1f}" for s in steps], rotation=45, fontsize=7)
    ax.set_yticklabels([f"{s:.1f}" for s in steps], fontsize=7)
    ax.set_title(f"$ w * x \quad (x_{pair[0]}, x_{pair[1]})$")
    ax.set_ylabel(f"$x_{pair[0]}$")
    ax.set_xlabel(f"$x_{pair[1]}$")
    plt.savefig("Weights" +  str(pair), dpi=400)
    plt.show()
