import numpy as np
example_1d_array=np.array([1,2,3,4,5,6,7,8,9,10])
print(example_1d_array[::2])
for i, step in enumerate(example_1d_array[::2]): 
    print(i,step)
