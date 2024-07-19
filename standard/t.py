import numpy as np
data=np.array([1,2,3,4,5,6,7])
data=data[::2]

for i, step in enumerate(data):
    if i == len(data) - 1:
        break  
    next_action = data[i + 1]
    state=step
    print(state,next_action)
