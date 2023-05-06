import wfdb
import matplotlib.pyplot as plt
import numpy as np


def add_noise(signal, amplitude):
    noise = np.random.normal(scale=amplitude, size=len(signal))
    return signal + noise


database_path = './ECG1/'
record = wfdb.rdrecord(database_path + 'Person_01/rec_1')
annotation = wfdb.rdann(database_path + 'Person_01/rec_1', 'atr')

noisy_signal = add_noise(record.p_signal[:, 0], 0.05)

# print('INFORMACJE O SYGNALE:', record.__dict__)
# print('INFORMACJE O ADNOTACJACH:', annotation.__dict__)


plt.plot(record.p_signal)
plt.xlabel('Czas [próbka]')
plt.ylabel('Napięcie [mV]')
plt.show()

plt.plot(noisy_signal)
plt.xlabel('Czas [próbka]')
plt.ylabel('Napięcie [mV]')
plt.show()

