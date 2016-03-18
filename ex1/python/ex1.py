import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt

data = []

with open('ex1data1.txt', 'rb') as csvfile:
    import csv
    datafile = csv.reader(csvfile, delimiter=',')
    for row in datafile:
        data.append(row)

data = np.array(data)
x = data[:,0]
y = data[:,1]
plt.plot(x, y, 'rx')
X = np.transpose([np.ones(x.shape[0]), x])
print(X)
