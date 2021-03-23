import core4
import core3
import matplotlib.pyplot as plt
import numpy as np

density = np.linspace(1e7,1e12, 1000)
ratio = []
counter = 0
for i in density:
    cor, re_cor = core3.solve(i, 1e7, 0.5, messages = False)
    cor_i, re_cor_i = core4.solve(i, 1e7, 0.5, messages = False)
    ratio.append(cor.pressure[0]/cor_i.pressure[0])
    cor = None
    re_cor = None
    cor_i = None
    re_cor_i = None
    counter = counter+1
    if counter%100 == 0:
        print(counter)

fig, ax = plt.subplots(dpi = 200)
ax.plot(density,ratio)
ax.set_xlabel('Density')
ax.set_ylabel('Ratio')
ax.set_title('Ratio of full non-relatistic to inverse integral')
ax.set_xscale('log')
ax.grid()
