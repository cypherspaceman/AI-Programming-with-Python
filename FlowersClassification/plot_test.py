import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [1, 2, 4, 8, 16]

fig, ax = plt.subplots()
ax.plot(x, y)

ax.set(xlabel='time (s)', ylabel='voltage (mV)', title='About as simple as it gets, folks')
# ax.grid()

# fig.savefig("test.png")
plt.show()
