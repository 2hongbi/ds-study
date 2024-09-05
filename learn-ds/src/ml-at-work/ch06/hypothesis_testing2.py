import numpy as np
import scipy
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt

# 테스트 데이터
# [지속 이용 전환 수, 이탈수]
a = [40, 165]
b = [62, 228]

print('표본 A: 크기={}, 전환 수={}, 평균={:.3f}'.format(sum(a), a[0], a[0] / sum(a)))
print('표본 B: 크기={}, 전환 수={}, 평균={:.3f}'.format(sum(b), b[0], b[0] / sum(b)))

x = np.linspace(0, 1, 200)

# 표본 A로부터 유입
n = sum(a)
p = a[0] / n
std = np.sqrt(p * (1 - p) / n)
y_a = scipy.stats.norm.pdf(x, p, std)

# 표본 B로부터 유입
n = sum(b)
p = b[0] / n
std = np.sqrt(p * (1 - p) / n)
y_b = scipy.stats.norm.pdf(x, p, std)

plt.figure(figsize=(7, 2))
plt.plot(x, y_a, label='표본 A')
plt.plot(x, y_b, label='표본 B')
plt.legend(loc='best')
plt.xlabel('신규 사용자 지속 이용 전환율')
plt.ylabel('가능성')
plt.show()
