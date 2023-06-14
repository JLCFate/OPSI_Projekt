import wfdb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA
import pywt
from sklearn.metrics import mean_squared_error, mean_absolute_error


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
    signal[:, ch] = noise

    return signal


def pca_denoise(signal_clear, signal_with_noise, n):
    # pca = PCA(n_components=n)
    # signal_clear_pca = pca.fit_transform(signal_clear)
    # signal_full_pca = np.hstack([signal_with_noise[:, 1:], signal_clear_pca])
    # pca_final = pca.fit_transform(signal_full_pca)

    pca = PCA(n_components=n)
    pca_final = pca.fit_transform(signal_with_noise)

    mse = mean_squared_error(signal_with_noise[:, 0], pca_final[:, 0])
    mae = mean_absolute_error(signal_with_noise[:, 0], pca_final[:, 0])
    print('MSE PCA: ', mse)
    print('MAE PCA:', mae)

    return pca_final


def ica_denoise(signal_clear, signal_with_noise, n):
    ica = FastICA(n_components=n, whiten='unit-variance')
    signal_clear_ica = ica.fit_transform(signal_clear)
    signal_full_ica = np.hstack([signal_with_noise[:, 1:], signal_clear_ica])
    ica_final = ica.fit_transform(signal_full_ica)

    # ica = FastICA(n_components=n, whiten='unit-variance')
    # ica_final = ica.fit_transform(signal_with_noise)
    # #Zmienic n na 4 i tez jest git

    mse = mean_squared_error(signal_with_noise[:, 0], ica_final[:, 0])
    mae = mean_absolute_error(signal_with_noise[:, 0], ica_final[:, 0])
    print('MSE ICA: ', mse)
    print('MAE ICA:', mae)

    return ica_final


def wavelet_denoise(signal, wavelet='db6', level=5):
    multi = 0.1
    values = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.median(np.abs(values[-level])) / multi
    for i in range(1, len(values)):
        values[i] = pywt.threshold(values[i], value=threshold, mode='soft')

    return pywt.waverec(values, wavelet)


def make_plot():
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
# get_info(record)
signal_noise = add_noise(record.p_signal, 0.2, ch=0)
signal_ref_ch_up = np.delete(record.p_signal, 0, axis=1)
signal_PCA = pca_denoise(signal_ref_ch_up, signal_noise, 3)
signal_ICA = ica_denoise(signal_ref_ch_up, signal_noise, 3)
signal_wavelet = wavelet_denoise(signal_noise)

mse3 = mean_squared_error(signal_noise[:, 0], signal_wavelet[:, 0])
mae3 = mean_absolute_error(signal_noise[:, 0], signal_wavelet[:, 0])
print('MSE wavelet: ', mse3)
print('MAE wavelet:', mae3)

make_plot()

# plt.plot(signal_ICA)
# # plt.set_ylabel('Amplitude')
# # plt.set_xlabel('Time')
# plt.suptitle("ECG after Wavelet transform")
# plt.tight_layout()
# plt.show()
