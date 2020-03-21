import numpy as np
x = np.loadtxt('Output_conv_1.txt')
x=x.reshape((-1,32,32,64))
print(x.shape)