import numpy as np
data = np.load('toy-ppi-feats.npy')
np.savetxt('toy-ppi-feats.csv', data, delimiter=',')
print(data)
