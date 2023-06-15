import wfdb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA
import pywt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def get_data(file_name='Norwegian/ath_001'):
    # wczytanie danych z PhysioNet zapisanych w lokalnym folderze

    database_path = './ECG1/'
    path = database_path + file_name
    ref = wfdb.rdrecord(path)
    rec = wfdb.rdrecord(path)

    return ref, rec


def get_info(rec):
    # wyświetlanie informacji na temat sygnalu
    # wykorzystywane glownie podczas wstepnej analizy sygnalu

    print('INFORMACJE O SYGNALE:', rec.__dict__)


def add_noise(signal, amplitude, ch=0):
    # dodanie szumu do sygnalu, mozliwosc manipulacji amplituda w celu testowania

    noise = np.random.normal(scale=amplitude, size=len(signal[:, ch]))
    signal[:, ch] += noise

    return signal


def pca_denoise(signal_clear, signal_with_noise, n):
    # funkcja realizujaca odszumianie z wykorzystaniem metody PCA
    # liczba komponentow dobierana wzgledem badanego sygnalu

    pca = PCA(n_components=n)
    pca_final = pca.fit_transform(signal_with_noise)

    # wyswietlenie matryk pozwalajacych na ocene sygnalu

    mse = mean_squared_error(signal_with_noise[:, 0], pca_final[:, 0])
    mae = mean_absolute_error(signal_with_noise[:, 0], pca_final[:, 0])
    print('MSE PCA: ', mse)
    print('MAE PCA:', mae)

    return pca_final


def pca_denoise_1(signal_clear, signal_with_noise, n):
    # alternatywne podejscie do wykonania odszumiania z wykorzystaniem metody PCA

    pca = PCA(n_components=n)
    signal_clear_pca = pca.fit_transform(signal_clear)
    signal_full_pca = np.hstack([signal_with_noise[:, 1:], signal_clear_pca])
    pca_final = pca.fit_transform(signal_full_pca)

    mse = mean_squared_error(signal_with_noise[:, 0], pca_final[:, 0])
    mae = mean_absolute_error(signal_with_noise[:, 0], pca_final[:, 0])
    print('MSE PCA: ', mse)
    print('MAE PCA:', mae)

    return pca_final


def ica_denoise(signal_clear, signal_with_noise, n):
    # funkcja realizujaca odszumianie z wykorzystaniem metody ICA
    # parametry dobierane wzgledem badanego sygnalu
    # whiten='unit-variance' - dodane w celu zniwelowania errora infoirmujacego o przyszlej zmianie domyslenj wartosci
    # whiten w wykorzystywanej bibliotece

    ica = FastICA(n_components=n, whiten='unit-variance')
    ica_final = ica.fit_transform(signal_with_noise)

    # wyswietlenie matryk pozwalajacych na ocene sygnalu

    mse = mean_squared_error(signal_with_noise[:, 0], ica_final[:, 0])
    mae = mean_absolute_error(signal_with_noise[:, 0], ica_final[:, 0])
    print('MSE ICA: ', mse)
    print('MAE ICA:', mae)

    return ica_final


def ica_denoise_1(signal_clear, signal_with_noise, n):
    # Alternatywne podejscie do odszumiania cygnaly z wykorzystaniem metody ICA

    ica = FastICA(n_components=n, whiten='unit-variance')
    signal_clear_ica = ica.fit_transform(signal_clear)
    signal_full_ica = np.hstack([signal_with_noise[:, 1:], signal_clear_ica])
    ica_final = ica.fit_transform(signal_full_ica)

    mse = mean_squared_error(signal_with_noise[:, 0], ica_final[:, 0])
    mae = mean_absolute_error(signal_with_noise[:, 0], ica_final[:, 0])
    print('MSE ICA: ', mse)
    print('MAE ICA:', mae)

    return ica_final


def wavelet_denoise(signal, wavelet='db6', level=5):
    # Funkcja realizujaca odszumianie z wykorzystaniem transformaty falkowej
    # Sygnal testowano na 2 falkach: db4 oraz db6
    # Zmienna multi odpowiada za modyfikowanie poziomu odszumiania

    multi = 0.1
    values = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.median(np.abs(values[-level])) / multi
    for i in range(1, len(values)):
        values[i] = pywt.threshold(values[i], value=threshold, mode='soft')

    return pywt.waverec(values, wavelet)


def make_plot():
    # Funkcja stworzona na potrzeby wizualizacji danych oraz pozniejszych wynikow
    # Tworzene jest 5 wykresow odpowiadajacych kolejno czytemu sygnalowi
    # sygnalowi zaszumionemu oraz wynikach odszumiania

    fig, (p) = plt.subplots(5, 1, figsize=(10, 10))
    for i in range(len(p)):
        title = ['Reference ECG Signal', 'Noisy ECG Signal', 'ECG after PCA()', 'ECG after ICA()',
                 'ECG after Wavelet transform']
        source = [referral.p_signal, signal_noise, signal_PCA, signal_ICA, signal_wavelet]
        p[i].plot(source[i][:, 0])
        p[i].set_ylabel('Amplitude')
        p[i].set_xlabel('Time')
        p[i].set_title(title[i])
    plt.tight_layout()
    plt.show()


# wywolanie wczesniej stworzonych funkcji oraz podanie im odpowiednich zmiennych
# sygnal wejsciowy zostal sklonowany w celu latwiejszej interpretacji w kodzie

referral, record = get_data()
signal_noise = add_noise(record.p_signal, 0.08, ch=0)
signal_ref_ch_up = np.delete(record.p_signal, 0, axis=1)
signal_PCA = pca_denoise(signal_ref_ch_up, signal_noise, 4)
signal_ICA = ica_denoise(signal_ref_ch_up, signal_noise, 3)
signal_wavelet = wavelet_denoise(signal_noise)

# Dodatkowe wyswietlenie informascji o sygnale po wykonaniu na nim odszumiania z wykorzystaniem transformaty falkowej

mse3 = mean_squared_error(signal_noise[:, 0], signal_wavelet[:, 0])
mae3 = mean_absolute_error(signal_noise[:, 0], signal_wavelet[:, 0])
print('MSE wavelet: ', mse3)
print('MAE wavelet:', mae3)

make_plot()

# Repozytorium projektu dostepne na serwisie GitHub : https://github.com/JLCFate/OPSI_Projekt
