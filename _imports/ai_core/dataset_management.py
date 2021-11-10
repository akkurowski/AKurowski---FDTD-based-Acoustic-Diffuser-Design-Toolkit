import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy as cp
from itertools import product
from scipy.signal import fftconvolve 
from numpy.fft import fft2
from scipy.interpolate import RectBivariateSpline
import numba as nb

# Funkcja pokazująca rozkład wartości funkcji nagrody dla
# losowo wygenerowanych geometrii w zbiorze wejściowym
def visualize_rewards(settings, preproc_data_path):
    preproc_data = np.load(preproc_data_path, allow_pickle=True)
    
    diff_coeffs   = []
    reward_values = []
    wdbnd_diff_values = []
    for dt_point in preproc_data:
        wdbnd_diff_values.append(dt_point['wideband_diff_coeffs'])
        diff_coeffs.append(get_bandpass_diff_coeffs_array(dt_point['bandpass_diff_coeffs']))
        reward_values.append(dt_point['reward_value'])
    
    plt.figure()
    plt.hist(reward_values,bins=60)
    plt.xlabel('wartość nagrody (średni wsp. dyfuzji - minimum między pł XY i YZ)')
    plt.ylabel('częstość występowania')
    plt.grid()
    plt.xlim([0,1])
    plt.title('szerokopasmowy wsp. nagrody')
    
    wideband_xy = []
    wideband_yz = []
    for entry in wdbnd_diff_values:
        wideband_xy.append(entry['plane_xy'])
        wideband_yz.append(entry['plane_yz'])
    
    plt.figure()
    plt.hist(wideband_xy,bins=60,label='plane xy')
    plt.hist(wideband_yz,bins=60,label='plane yz')
    plt.xlabel('wartość szerokopasmowego wsp. dyfuzji')
    plt.ylabel('częstość występowania')
    plt.grid()
    plt.xlim([0,1])
    plt.legend()
    plt.title('szerokopasmowy wsp. dyfuzji')
    
    diff_coeffs = np.stack(diff_coeffs)
    plt.figure()
    plt.subplot(2,1,1)
    for i, f0 in enumerate(settings['basic']['bands_f0']):
        plt.hist(diff_coeffs[:,0,i],bins=60, label=f'{f0} Hz, xy', alpha=0.8)
    plt.ylabel('częstość występowania (xy)')
    plt.legend()
    plt.grid()
    
    plt.subplot(2,1,2)
    for i, f0 in enumerate(settings['basic']['bands_f0']):
        if f0 != 1000: continue
        plt.hist(diff_coeffs[:,1,i],bins=60, label=f'{f0} Hz, yz', alpha=0.8)
    plt.xlabel('wartość nagrody (pasmowy wsp. dyfuzji pł yz)')
    plt.ylabel('częstość występowania (yz)')
    plt.legend()
    plt.grid()
    plt.xlim([0,1])
    plt.suptitle('Pasmowe współczynniki nagrody')

def augmentation_procedure(pattern, bandpass_diff_coeffs, mirroring, rotation, n_steps_x_shift, shift_step=0.15):
    # Lokalna kopia struktury danych przechowującej kształt
    local_pattern_copy = cp.deepcopy(pattern)
    bandpass_df_copy   = cp.deepcopy(bandpass_diff_coeffs)
    
    # -----------------
    # Generowanie lustrzanych wersji patternu
    # (współczynniki dyfuzji są niezmienne względem 
    # lustrzanej zmiany geometrii dyfuzora)
    if mirroring[0] == 1:
        local_pattern_copy = np.fliplr(local_pattern_copy)
    if mirroring[1] == 1:
        local_pattern_copy = np.flipud(local_pattern_copy)
    
    # -----------------
    # Obracanie dyfuzora względem osi na której propaguje
    # się pobudzenie, współczynniki ulegają zamianie kolejnością
    # przy każdej zmianie kolejności
    if rotation in [0,2]: # 0 i 180 st., nic się nie zmienia
        pass
    elif rotation in [1,3]: # 90 i 270 st., trzeba zamienić współczynniki kolejnością!
        bandpass_df_copy = bandpass_df_copy[[1,0],:]
    
    local_pattern_copy = np.rot90(local_pattern_copy,rotation)
    
    # -----------------
    # Dodatkowo możemy pattern przesunąć "do przodu lub do tyłu",
    # i nie powinno mieć to znaczenia dla działania dyfuzora
    # (w pomiarze można taki dyfuzor "cofnąć" o wartość przesunięcia i uzyskać
    # identyczny wynik.
    local_pattern_copy += shift_step*n_steps_x_shift
    
    return local_pattern_copy, bandpass_df_copy

def augment_by_mirroring(patterns, bandpass_diff_coeffs):
    output_patterns = []
    output_bdc      = []
    for pattern in patterns:
        pattern_lr   = np.fliplr(cp.deepcopy(pattern))
        pattern_ud   = np.flipud(cp.deepcopy(pattern))
        pattern_lrud = np.fliplr(np.flipud(cp.deepcopy(pattern)))
        output_patterns += [pattern, pattern_lr, pattern_ud, pattern_lrud]
        if type(bandpass_diff_coeffs) != list:
            output_bdc += [bandpass_diff_coeffs]*4
        else:
            for i in range(4): output_bdc += bandpass_diff_coeffs
    
    return output_patterns, output_bdc

def augment_by_rotating(patterns,bandpass_diff_coeffs):
    output_patterns = []
    output_bdc      = []
    for pattern, bdc in zip(patterns,bandpass_diff_coeffs):
        pattern_90  = np.rot90(cp.deepcopy(pattern),1)
        pattern_180 = np.rot90(cp.deepcopy(pattern),2)
        pattern_270 = np.rot90(cp.deepcopy(pattern),3)
        output_patterns += [pattern, pattern_90, pattern_180, pattern_270]

        output_bdc.append(bdc)
        output_bdc.append(bdc[[1,0]])
        output_bdc.append(bdc)
        output_bdc.append(bdc[[1,0]])
    
    return output_patterns, output_bdc

# Augmentacja przykładu ze zbioru danych:
# zbiór jest rozszerzany o przykłady będące odbiciami lustrzanymi, 
# obracane o 90 stopni i przesuwane "do przodu"
def get_aug_exmaples_package(pattern, bandpass_diff_coeffs):
    
    if type(bandpass_diff_coeffs) != np.ndarray:
        raise RuntimeError(f'wsp. dyfuzji na wejściu augmentacji muszą mieć typ np.array, podany typ: {type(bandpass_diff_coeffs)}')
    
    X = [pattern]
    Y = [bandpass_diff_coeffs]
    augmentation_info = []
    
    X, Y = augment_by_mirroring(X,Y)
    X, Y = augment_by_rotating(X,Y)
    
    return X, Y, augmentation_info

def get_bandpass_diff_coeffs_array(bandpass_diff_coeffs):
        bdc_mtx = [
            list(bandpass_diff_coeffs['plane_xy'].values()),
            list(bandpass_diff_coeffs['plane_yz'].values())]
        bdc_mtx = np.array(bdc_mtx)
        return bdc_mtx

def norm(X): 
    return X - np.mean(X)

def single_shot_schroeder_parameterize(x):
    x_norm     = norm(x)
    x_spectrum = np.abs(fft2(x_norm))
    x_cepstrum = np.abs(fft2(norm(x_spectrum)))
    
    base_indices_x        = np.arange(x.shape[0])
    base_indices_y        = np.arange(x.shape[1])
    autocorr_indices_x    = np.linspace(0,x.shape[0],2*x.shape[0]-1)
    autocorr_indices_y    = np.linspace(0,x.shape[1],2*x.shape[1]-1)
    
    orig_autocorr         = fftconvolve(x,np.flip(x,[0,1]))
    autocorr_interpolator = RectBivariateSpline(autocorr_indices_x,autocorr_indices_y,orig_autocorr,kx=5,ky=5)
    x_autocorr            = autocorr_interpolator(base_indices_x,base_indices_y)
    
    x_parameterized = np.stack([x,x_spectrum,x_cepstrum,x_autocorr],axis=2)
    return x_parameterized

# funkcja do ekstracji kanału autokorelacji z wymiarów dyfuzora
def extract_autocorrelations(X):
    base_indices_x     = np.arange(X.shape[1])
    base_indices_y     = np.arange(X.shape[2])
    autocorr_indices_x = np.linspace(0,X.shape[1],2*X.shape[1]-1)
    autocorr_indices_y = np.linspace(0,X.shape[2],2*X.shape[2]-1)
    
    X_interp = np.zeros_like(X)
    print()
    print('obliczanie parametrów autokorelacyjnych')
    for ex_num in tqdm(range(X.shape[0])):
        orig_autocorr          = fftconvolve(X[ex_num,:,:],np.flip(X[ex_num,:,:],[0,1]))
        autocorr_interpolator  = RectBivariateSpline(autocorr_indices_x,autocorr_indices_y,orig_autocorr,kx=5,ky=5)
        interp_autocorr        = autocorr_interpolator(base_indices_x,base_indices_y)
        X_interp[ex_num,:,:]   = interp_autocorr
    print()
    
    return X_interp
    
# Funkcja przygotowująca dane (skonwertowane do postaci geometria --> wsp. dyfuzji)
# do podania na wejście aproksymującej sieci neuronowej
def extract_datasets(settings, preproc_data_path, input_sets_path):

    # Odczyt danych przetworzonych przez poprzednie polecenie konwertujące
    preproc_data = np.load(preproc_data_path, allow_pickle=True)
    
    # Zablokowanie generatora - aby sieci nigdy nie ujrzały zbioru testowego
    np.random.seed(30012021)
    
    # "Przemieszanie" zbioru danych przed podziałem na podzbiory
    np.random.shuffle(preproc_data)
    
    # Wyliczenie rozmiarów zbiorów danych 
    num_examples         = len(preproc_data)
    train_val_test_split = settings['basic']['train_val_test_split']
    ttv_split_sum        = np.sum(train_val_test_split)
    
    train_indices      = [0,train_val_test_split[0]]
    validation_indices = [train_indices[1],train_indices[1]+train_val_test_split[1]]
    test_indices       = [validation_indices[1],validation_indices[1]+train_val_test_split[2]]
    
    train_indices      = (np.array(train_indices)/ttv_split_sum*num_examples).astype('int')
    validation_indices = (np.array(validation_indices)/ttv_split_sum*num_examples).astype('int')
    test_indices       = (np.array(test_indices)/ttv_split_sum*num_examples).astype('int')
    
    # Podzielenie danych na podzbiory wejściowe (X) i wyjściowe (Y).
    input_datasets_X = {}
    input_datasets_Y_diff_coeffs = {}
    input_datasets_Y_reward = {}
    input_datasets_Y_wideband = {}
    
    i=-1
    for dt_point in tqdm(preproc_data):
        i += 1
        # Sprawdzamy, w którym zakresie leży wiersz danych
        # i przydzielamy wiersz do odpowiadającego temu 
        # zakresowi podzbioru danych.
        if   train_indices[0] <= i < train_indices[1]:
            set_name = 'training'
        elif validation_indices[0] <= i < validation_indices[1]:
            set_name = 'validation'
        elif test_indices[0] <= i < test_indices[1]:
            set_name = 'test'
        else:
            raise RuntimeError('niepoprawny indeks zbioru w procedurze ekstrakcji zbiorów danych')
        
        # Jeśli słownik zbiorczy jeszcze nie posiada 
        # odpowiednich kluczy dla podzbiorów, to
        # tutaj je utworzymy
        if set_name not in input_datasets_X.keys():
            input_datasets_X.update({set_name:[]})
        if set_name not in input_datasets_Y_diff_coeffs.keys():
            input_datasets_Y_diff_coeffs.update({set_name:[]})
        if set_name not in input_datasets_Y_reward.keys():
            input_datasets_Y_reward.update({set_name:[]})
        if set_name not in input_datasets_Y_wideband.keys():
            input_datasets_Y_wideband.update({set_name:[]})
        
        # Wydzielenie odpowiednich pól danych i przydział do
        # odpowiednich podzbiorów.
        pattern_dims_meters   = dt_point['pattern_dims_meters']
        bandpass_diff_coeffs  = dt_point['bandpass_diff_coeffs']
        reward_value          = dt_point['reward_value']
        wideband_diff_coeff   = dt_point['wideband_diff_coeffs']
        
        # Konwersja słownika wsp. dyfuzji na format macierzowy
        bdc_mtx = get_bandpass_diff_coeffs_array(bandpass_diff_coeffs)
        
        # Augmentacja zbioru danych - zbiór jest rozszerzany o 
        # przykłady będące odbiciami lustrzanymi, obracane 
        # o 90 stopni i przesuwane "do przodu"
        augmentation = settings['basic']['augment_training_set']
        if augmentation:
            X_aug, Y_aug_diffc, _ = get_aug_exmaples_package(pattern_dims_meters, bdc_mtx)
            input_datasets_X[set_name]             += X_aug
            input_datasets_Y_diff_coeffs[set_name] += Y_aug_diffc
            input_datasets_Y_reward[set_name]      += [reward_value]*len(Y_aug_diffc)
            input_datasets_Y_wideband[set_name]    += [wideband_diff_coeff['plane_xy']]*len(Y_aug_diffc)
        else:
            input_datasets_X[set_name]             += [pattern_dims_meters]
            input_datasets_Y_diff_coeffs[set_name] += [bdc_mtx]
            input_datasets_Y_reward[set_name]      += [reward_value]
            input_datasets_Y_wideband[set_name]    += [wideband_diff_coeff['plane_xy']]
            
    # Konwersja danych rozdzielonych w poprzednim kroku na format np.ndarray.
    for input_set in [
        input_datasets_X,
        input_datasets_Y_diff_coeffs,
        input_datasets_Y_reward,
        input_datasets_Y_wideband]:
        for subset_name in input_set.keys():
            input_set[subset_name] = np.stack(input_set[subset_name])
    
    # Podzbiory wejściowe zostaną dodatkowo sparametryzowane:
    def parameterize_X(X):
        
        X_norm     = norm(X)
        X_spectrum = np.abs(fft2(X_norm,axes=[1,2]))
        X_cepstrum = np.abs(fft2(norm(X_spectrum),axes=[1,2]))
        X_autcorr  = extract_autocorrelations(X_norm)
        
        X          = X[:,:,:,np.newaxis]
        X_spectrum = X_spectrum[:,:,:,np.newaxis]
        X_cepstrum = X_cepstrum[:,:,:,np.newaxis]
        X_autcorr  = X_autcorr[:,:,:,np.newaxis]
        
        # X          = np.concatenate([X, X_spectrum, X_cepstrum, X_autcorr], axis=3)
        X          = np.concatenate([X_spectrum, X_autcorr], axis=3)
        
        return X
    
    print()
    print('Parametryzacja zbiorów wejściowych:')
    multichannel_input_datasets_X = {}
    print('\ttreningowy')
    multichannel_input_datasets_X.update({'training':   parameterize_X(input_datasets_X['training'])})
    print('\twalidacyjny')
    multichannel_input_datasets_X.update({'validation': parameterize_X(input_datasets_X['validation'])})
    print('\ttestowy')
    multichannel_input_datasets_X.update({'test':       parameterize_X(input_datasets_X['test'])})
    
    input_datasets = {
        'X':multichannel_input_datasets_X,
        'Y_diff_coeffs':input_datasets_Y_diff_coeffs,
        'Y_reward':input_datasets_Y_reward,
        'Y_wideband':input_datasets_Y_wideband
        }
        
    print()
    print('rozmiary zbiorów wejściowych (X):')
    print(f"\t treningowy:  {input_datasets['X']['training'].shape}")
    print(f"\t walidacyjny: {input_datasets['X']['validation'].shape}")
    print(f"\t testowy:     {input_datasets['X']['test'].shape}")
    print()
    print('rozmiary zbiorów wyjściowych w wariancie macierzy współczynników dyfuzji(Y_diff_coeffs):')
    print(f"\t treningowy:  {input_datasets['Y_diff_coeffs']['training'].shape}")
    print(f"\t walidacyjny: {input_datasets['Y_diff_coeffs']['validation'].shape}")
    print(f"\t testowy:     {input_datasets['Y_diff_coeffs']['test'].shape}")
    print()
    print()
    print('rozmiary zbiorów wyjściowych w wariancie wartości funkcji nagrody (Y_reward):')
    print(f"\t treningowy:  {input_datasets['Y_reward']['training'].shape}")
    print(f"\t walidacyjny: {input_datasets['Y_reward']['validation'].shape}")
    print(f"\t testowy:     {input_datasets['Y_reward']['test'].shape}")
    print()
    
    # zapis danych na dysk
    np.save(input_sets_path, input_datasets)