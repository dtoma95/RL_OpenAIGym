import numpy as np
f = open("data", "r")
lines = f.readlines()
train = []
for line in lines:
    splitted = line.split(" ")
    train.append(splitted[-1])
    if(len(train)>1000):
        break
# summarize history for accuracy
import matplotlib.pyplot as plt
train = np.array(train, dtype=float)
plt.plot(train)
plt.ylabel('steps')
plt.xlabel('episode')
plt.show()