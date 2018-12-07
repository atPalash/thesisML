import numpy as np

labels = [0, 1, 5, 6]
image_names = ['objB', 'objA', 'objD', 'objC']
DAT = np.column_stack((image_names, labels))
np.savetxt('file_numpy.txt', DAT, delimiter=" ", fmt="%s")

# x = y = z = np.arange(0.0,5.0,1.0)
# np.savetxt('test.out', x, delimiter=',')   # X is an array

