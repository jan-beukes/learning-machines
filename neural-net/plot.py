import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("train.txt")
epochs = data[:,0]
validation_acc = data[:,1]
train_acc = data[:,2]

plt.plot(epochs, validation_acc, epochs, train_acc)
plt.title("Training Performance")
plt.legend(["validation set", "training set"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
