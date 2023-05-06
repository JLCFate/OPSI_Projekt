import wfdb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def add_noise(signal, amplitude):
    noise = np.random.normal(scale=amplitude, size=len(signal))
    return signal + noise


database_path = './ECG1/'
record = wfdb.rdrecord(database_path + 'Person_01/rec_1')
annotation = wfdb.rdann(database_path + 'Person_01/rec_1', 'atr')

signal_noise = add_noise(record.p_signal[:, 0], 0.05)

# print('INFORMACJE O SYGNALE:', record.__dict__)
# print('INFORMACJE O ADNOTACJACH:', annotation.__dict__)

pca = PCA(n_components=3)
pca.fit(signal_noise.reshape(-1, 1))
signal_PCA = pca.inverse_transform(pca.transform(signal_noise.reshape(-1, 1))).flatten()

fig, (p1, p2, p3) = plt.subplots(3, 1, figsize=(10, 8))

p1.plot(record.p_signal)
p1.set_ylabel('Amplituda')
p1.set_title('Oryginalny sygnał EKG')

p2.plot(signal_noise)
p2.set_ylabel('Amplituda')
p2.set_title('Sygnał EKG + szum')

p3.plot(signal_PCA)
p3.set_xlabel('Czas [próbka]')
p3.set_ylabel('Amplituda')
p3.set_title('Sygnał EKG po PCA()')

plt.tight_layout()
plt.show()
