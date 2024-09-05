import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

x = np.arange(0, 21)
y = scipy.stats.binom.pmf(x, 20, 0.5) # 이항분포 (20, 0.5)를 따름

plt.figure(figsize=(8, 2))
plt.bar(x, y)
plt.xlabel('앞면이 나온 횟수')
plt.ylabel('확률')