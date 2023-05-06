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


fig, (p1, p2, p3, p4, p5) = plt.subplots(5, 1, figsize=(10, 10))

p1.plot(record.p_signal)
p1.set_ylabel('Amplitude')
p3.set_xlabel('Time')
p1.set_title('Reference ECG Signal')

p2.plot(signal_noise)
p3.set_xlabel('Time')
p2.set_ylabel('Amplitude')
p2.set_title('Noisy ECG Signal')

p3.plot(signal_PCA)
p3.set_xlabel('Time')
p3.set_ylabel('Amplitude')
p3.set_title('ECG after PCA()')

p4.plot(signal_ICA)
p4.set_xlabel('Time')
p4.set_ylabel('Amplitude')
p4.set_title('ECG after ICA()')

p5.plot(signal_ICA)
p5.set_xlabel('Time')
p5.set_ylabel('Amplitude')
p5.set_title('ECG after Wavelet transform')

plt.tight_layout()
plt.show()
