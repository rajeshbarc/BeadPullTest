import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-white')
cavityDiameter = 160e-3
zAxisBead = np.linspace(-cavityDiameter / 2, cavityDiameter / 2, 500)
data = pd.read_csv('data.csv', header=None)
eFieldBead = data[0].to_numpy()
eFieldBead[:249] = - eFieldBead[:249]
data = pd.read_csv('simulated.csv', header=None)
zAxisSimulated = data[0].to_numpy()
zAxisSimulated = zAxisSimulated * 1e-3
eFieldSimulated = data[1].tolist()
# plt.plot(zAxisBead, eFieldBead)
# plt.plot(zAxisSimulated, eFieldSimulated)

frequency = 148.009205e6
omega = 2 * np.pi * frequency
beta = np.linspace(0.03, 0.2, 1000)
vRF = np.array([])
for i in beta:
    vRF = np.append(vRF, np.trapz(eFieldBead * np.sin((omega * zAxisBead) / (i * 3e8)), zAxisBead))
vDC = np.trapz(np.abs(eFieldBead), zAxisBead)
ttf = vRF / vDC
plt.plot(beta, ttf, color='cornflowerblue', label='Bead-Pull')
max_value = np.max(ttf)
max_index = np.argmax(ttf)
plt.plot([beta[max_index], beta[max_index]], [0, max_value], linestyle='--', color='cornflowerblue')
print(beta[max_index])
vRF = np.array([])
for i in beta:
    vRF = np.append(vRF, np.trapz(eFieldSimulated * np.sin((omega * zAxisSimulated) / (i * 3e8)), zAxisSimulated))
vDC = np.trapz(np.abs(eFieldSimulated), zAxisSimulated)
ttf = vRF / vDC
plt.plot(beta, ttf, color='orange', label='Simulated')
max_value = np.max(ttf)
max_index = np.argmax(ttf)
plt.plot([beta[max_index], beta[max_index]], [0, max_value], linestyle='--', color='orange')
plt.locator_params(axis='y', nbins=10)
plt.locator_params(axis='x', nbins=20)
plt.ylim(0, None)
plt.legend(prop={'size': 18})
plt.xlabel("Beta", fontsize=18)
plt.ylabel("TTF", fontsize=18)
print(beta[max_index])
plt.show()
