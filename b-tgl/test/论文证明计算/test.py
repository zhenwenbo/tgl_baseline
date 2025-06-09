

import math
import numpy as np

def kl(a, B, V):
    return a * (a * V + a * V * B + 1 - a) * math.log(a * V + a * V * B + 1 - a) + math.pow(1-a,2)


a1 = np.array(range(0, 10)) / 10
b1 = [60000]
v1 = [1981,20000000]
for v in v1:
    for b in b1:
        for a in a1:
            print(f"|V|:{v} |B|:{b} alpha:{a} res:{kl(a,b,v)}")
