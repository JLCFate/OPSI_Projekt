import wfdb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA
import pywt


def get_data(file_name='Norwegian/ath_001'):
    database_path = './ECG1/'
    path = database_path + file_name
    ref = wfdb.rdrecord(path)
    rec = wfdb.rdrecord(path)

    return ref, rec


def get_info(rec):
    print('INFORMACJE O SYGNALE:', rec.__dict__)


def add_noise(signal, amplitude, ch=0):
    noise = np.random.normal(scale=amplitude, size=len(signal[:, ch]))
    signal[:, ch] += noise

    return signal


def pca_denoise(signal_clear, signal_with_noise, n):
    pca = PCA(n_components=n)
    signal_clear_pca = pca.fit_transform(signal_clear)
    signal_full_pca = np.hstack([signal_with_noise[:, 1:], signal_clear_pca])
    pca_final = pca.fit_transform(signal_full_pca)

    return pca_final


def ica_denoise(signal_clear, signal_with_noise, n):
    ica = FastICA(n_components=n)
    signal_clear_ica = ica.fit_transform(signal_clear)
    signal_full_ica = np.hstack([signal_with_noise[:, 1:], signal_clear_ica])
    ica_final = ica.fit_transform(signal_full_ica)

    return ica_final


def wavelet_denoise(signal, wavelet='db4', level=2):

    values = pywt.wavedec(signal, wavelet, level=level)

    threshold = np.median(np.abs(values[-level])) / 0.6 #0.6745
    values[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in values[1:])

    return pywt.waverec(values, wavelet)


def make_plot():
    p = ['p1', 'p2', 'p3', 'p4', 'p5']

    fig, (p) = plt.subplots(5, 1, figsize=(10, 10))
    for i in range(len(p)):
        title = ['Reference ECG Signal', 'Noisy ECG Signal', 'ECG after PCA()', 'ECG after ICA()',
                 'ECG after Wavelet transform']
        source = [referral.p_signal, signal_noise, signal_PCA, signal_ICA, signal_wavelet]
        p[i].plot(source[i][:, 0])
        p[i].set_ylabel('Amplitude')
        p[i].set_xlabel('Time')
        if i == 2:
            p[i].invert_yaxis()  # vertical flip
        p[i].set_title(title[i])
    plt.tight_layout()
    plt.show()


referral, record = get_data()
get_info(record)
signal_noise = add_noise(record.p_signal, 0.25, ch=0)
signal_ref_ch_up = np.delete(record.p_signal, 0, axis=1)
signal_PCA = pca_denoise(signal_ref_ch_up, signal_noise, 3)
signal_ICA = ica_denoise(signal_ref_ch_up, signal_noise, 3)


signal_wavelet = wavelet_denoise(signal_noise)

make_plot()

