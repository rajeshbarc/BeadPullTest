import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scienceplots

plt.style.use(['science', 'ieee', 'no-latex', 'vibrant'])
matplotlib.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)

frequency = 148.5e6
omega = 2 * np.pi * frequency
df = pd.read_csv('z_Ez.csv', sep=',', header=None)
df1 = pd.read_excel('capacitance_and_peak.xlsx', header=None)
gap = np.array([30, 35, 40, 45, 50])
e_l = np.array([40, 40, 40, 40, 40])
capacitance = df1[0].to_numpy()
peakE = df1[1].to_numpy()
# Assuming each row has exactly two numbers separated by ';'
# Extract the numbers into two separate lists
z = df[0].to_numpy()
ez = df[1].to_numpy()
sign_changes = np.where(np.diff(np.signbit(z)))[0] + 1
zAxisArray = []
eZArray = []
index = 0
for i in sign_changes[1::2]:
    zAxisArray.append(z[index:i])
    eZArray.append(ez[index:i])
    index = i
zAxisArray.append(z[index:])
eZArray.append(ez[index:])
beta = np.linspace(0.03, 0.25, 1000)
betaOptimum = []
eAcc = []
for k in range(0, len(zAxisArray)):
    vRF = np.array([])
    for i in beta:
        vRF = np.append(vRF,
                        np.trapz(eZArray[k] * np.sin((omega * zAxisArray[k] * 1e-3) / (i * 3e8)), zAxisArray[k] * 1e-3))
        vDC = np.trapz(np.abs(eZArray[k]), zAxisArray[k] * 1e-3)
    ttf = vRF / vDC
    max_value = np.max(vRF)
    max_index = np.argmax(ttf)
    betaOptimum.append(beta[max_index] * 100)
    eAcc.append(max_value / (2e-3 * (gap[k] + e_l[k])))
print(eAcc)
fieldRatio = np.divide(peakE, np.array(eAcc))
fig, ax1 = plt.subplots()
ax1.plot(gap, betaOptimum, marker='*', markersize=6)
ax1.set_ylabel(r'$\beta_{opt}$ [%]', labelpad=0.3)
ax1.set_xlabel("Gap [mm]", labelpad=0.3)
ax2 = ax1.twinx()
ax2.plot(gap, fieldRatio, color='blue', marker='o', markersize=6)
ax2.set_ylabel("Peak E ratio", labelpad=0.3)
plt.tight_layout()
plt.show()
