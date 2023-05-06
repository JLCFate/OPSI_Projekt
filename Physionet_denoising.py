import wfdb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA
import pywt


def add_noise(signal, amplitude):
    noise = np.random.normal(scale=amplitude, size=len(signal))
    return signal + noise


def wavelet_denoise(signal, wavelet='db4', level=1):

    values = pywt.wavedec(signal, wavelet, level=level)

    threshold = np.median(np.abs(values[-level])) / 0.6 #0.6745
    values[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in values[1:])

    return pywt.waverec(values, wavelet)


database_path = './ECG1/'
record = wfdb.rdrecord(database_path + 'Person_01/rec_1')
annotation = wfdb.rdann(database_path + 'Person_01/rec_1', 'atr')

signal_noise = add_noise(record.p_signal[:, 0], 0.05)

# print('INFORMACJE O SYGNALE:', record.__dict__)
# print('INFORMACJE O ADNOTACJACH:', annotation.__dict__)


pca = PCA(n_components=1)
pca.fit(signal_noise.reshape(-1, 1))
signal_PCA = pca.inverse_transform(pca.transform(signal_noise.reshape(-1, 1))).flatten()


ica = FastICA(n_components=1)
signal_ICA = ica.fit_transform(signal_noise.reshape(-1, 1)).flatten()

signal_wavelet = wavelet_denoise(signal_noise)

p = ['p1', 'p2', 'p3', 'p4', 'p5']

fig, (p) = plt.subplots(5, 1, figsize=(10, 10))


for i in range(len(p)):
    title = ['Reference ECG Signal', 'Noisy ECG Signal', 'ECG after PCA()', 'ECG after ICA()', 'ECG after Wavelet transform']
    source = [record.p_signal, signal_noise, signal_PCA, signal_ICA, signal_wavelet]
    p[i].plot(source[i])
    p[i].set_ylabel('Amplitude')
    p[i].set_xlabel('Time')
    p[i].set_title(title[i])


plt.tight_layout()
plt.show()
