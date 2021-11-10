# ---------------------------------------------------- #
# Główny skrypt odpowiedzialny za bazujące na AI
# projektowanie geometrii dyfuzorów akustycznych
# autor:  Adam Kurowski
# data:   29.01.2021
# e-mail: akkurowski@gmail.com
# ---------------------------------------------------- #

from _imports import *
import numpy as np
import shutil
import os
from tqdm import tqdm

if os.name == 'nt':
    os.system('cls')
else:
    os.system('clear')

# -------------------------------------------------------
# Odczyt nastaw z pliku konfiguracyjnego
# -------------------------------------------------------

# Odczyt pliku konfiguracyjnego.
CONFIG_PATH = '_settings/ai_default.ini'
settings    = read_config(CONFIG_PATH)

# -------------------------------------------------------
# Wykonanie pliku
# -------------------------------------------------------

# Zablokowanie generatora losowego
if settings['basic']['lock_random_generator']:
    np.random.seed(10012021)

# Narysowanie nagłówka programu
print()
print('--------------------------------------------------')
print(' Skrypt AI generujący geometrie dyfuzorów poprzez')
print(' analizę symulacji uzyskanych metodą FDTD')
print(' autor:         A. Kurowski')
print(' e-mail:        akkurowski@gmail.com')
print(' wersja z dnia: 29.01.2021')
print('--------------------------------------------------')
print()

impres_dataset_path = settings['basic']['impres_dataset_path']
reference_file_path = settings['basic']['reference_file_path']

reference_data = None
try:
    reference_data = np.load(reference_file_path, allow_pickle=True).item()
except:
    print(f'nie udało się załadować pliku z danymi referencyjnymi ({reference_file_path})')

preprocessed_data_path = '_preprocessed_ai_data/processed_data.npy'
input_sets_path        = '_preprocessed_ai_data/input_sets.npy'

# Uruchomienie głównej pętli interakcji
while True:
    main_menu_items = []
    main_menu_items.append('przetwórz wstępnie zbiór danych')
    main_menu_items.append('wydziel zbiory treningowe (treningowy, walidacyjny, testowy)')
    main_menu_items.append('trenuj aproksymator')
    main_menu_items.append('załaduj ponownie ustawienia skryptu')
    # main_menu_items.append('zwizualizuj przykład działania augmentacji danych')
    main_menu_items.append('zwizualizuj histogram sygnału nagrody')
    if settings['basic']['approximator_nn_type'] != 'scalar_reward':
        main_menu_items.append('zwizualizuj przykład działania aproksymatora')
        main_menu_items.append('zwizualizuj odpowiedzi sieci i ground truth na wykresie pudełkowym')
    main_menu_items.append('wyjście')
    
    mm_ans = ask_user_for_an_option_choice('\nWybierz nr akcji do wykonania', 'Nr akcji:', main_menu_items)
    
    if mm_ans == 'przetwórz wstępnie zbiór danych':
        
        print()
        if reference_data is None:
            print('Nie udało się załadować danych referencyjnych, co uniemożliwia przetworzenie zbioru.')
            print('Sprawdź, czy plik znajduje się pod ścieżką wskazaną w pliku konfiguracyjnym i uruchom skrypt jeszcze raz.')
            continue
        
        filter_defs = get_filter_defs(settings)
        
        print()
        print('Przetwarzanie wstępne w toku, proszę czekać...')
        print()
        
        already_processed_files = []
        if os.path.isfile(preprocessed_data_path):
            already_processed_data = np.load(preprocessed_data_path, allow_pickle=True).tolist()
            for row in already_processed_data:
                if 'file_name' not in list(row.keys()):
                    already_processed_files = []
                    break
                already_processed_files.append(row['file_name'])
        
            processed_geom_output = already_processed_data
        else:
            processed_geom_output = []
        
        for fname in tqdm(os.listdir(impres_dataset_path)):
            
            if fname in already_processed_files:
                continue
            
            fpath    =  os.path.join(impres_dataset_path,fname)
            if fpath == reference_file_path: continue
            
            shape_data   = np.load(fpath, allow_pickle=True).item()
            fs_diff = shape_data['fs']
            if fs_diff != settings['basic']['fs']:
                raise RuntimeError(f"Dane dyfuzora mają częstotliwość próbkowania ({fs_diff}) inną niż częstotliwość skryptu ({settings['fs']})")
            
            wideband_diff_coeffs, bandpass_diff_coeffs, reward_value = diffusion_reward_value(settings, reference_data, shape_data, filter_defs = filter_defs)
            
            pattern_meters = shape_data['pattern']/shape_data['num_element_height_levels']*shape_data['diffuser_depth']
            
            processed_geom_output.append({
                'file_name':fname,
                'pattern_dims_meters':pattern_meters,
                'bandpass_diff_coeffs':bandpass_diff_coeffs,
                'wideband_diff_coeffs':wideband_diff_coeffs,
                'reward_value':reward_value})
        
        print()
        print(f'przetwarzanie zakończone, wynik zapisany pod ścieżką: {preprocessed_data_path}')
        np.save(preprocessed_data_path,processed_geom_output)
        
    elif mm_ans == 'załaduj ponownie ustawienia skryptu':
        settings    = read_config(CONFIG_PATH)
        if settings['basic']['lock_random_generator']:
            np.random.seed(10012021)
        impres_dataset_path = settings['basic']['impres_dataset_path']
        reference_file_path = settings['basic']['reference_file_path']
        print('\nustawienia załadowano poprawnie\n')
        
    elif mm_ans == 'zwizualizuj histogram sygnału nagrody':
        visualize_rewards(settings, preprocessed_data_path)
        plt.show()
    
    elif mm_ans == 'zwizualizuj przykład działania augmentacji danych':
        if not os.path.isfile(preprocessed_data_path):
            print(f'Nie znaleziono przetworzonych wstępnie danych, które potrzebne są do wizualizacji')
            print(f'ścieżka pod którą szukane są dane: {preprocessed_data_path}')
            continue
        
        list_of_examples     = np.load(preprocessed_data_path, allow_pickle=True)
        random_example       = list_of_examples[[np.random.randint(0,len(list_of_examples))]][0]
        
        pattern_dims_meters  = random_example['pattern_dims_meters']
        bandpass_diff_coeffs = random_example['bandpass_diff_coeffs']
        reward_value         = random_example['reward_value']
        bdc_mtx              = get_bandpass_diff_coeffs_array(bandpass_diff_coeffs)
        
        X_aug, Y_aug_diffc, augmentation_info = get_aug_exmaples_package(pattern_dims_meters, bdc_mtx)
        
        for X, Y, aug_info in zip(X_aug, Y_aug_diffc, augmentation_info):
            print('schemat rozszerzenia:')
            print('\todb. lustrzane: ',aug_info[0])
            print('\tobrót:          ',aug_info[1])
            print('\tprzesunięcie:   ',aug_info[2])
            print('\tnagroda:', reward_value)
            print()
            print('macierz definicji dyfuzora:')
            print(X)
            print()
            print('macierz współczynników odpowiedzi:')
            print(Y)
            print()
            print()
        
            if not ask_for_user_preference('Czy kontynuować wizualizowanie działania mechanizmu augmentacji?'):
                break
    
    elif mm_ans == 'zwizualizuj przykład działania aproksymatora':
        draw_approximator_example(settings, input_sets_path)
        plt.show()
    
    elif mm_ans == 'zwizualizuj odpowiedzi sieci i ground truth na wykresie pudełkowym':
        draw_pred_and_val_boxplot(settings, input_sets_path)
        plt.show()
        
    elif mm_ans == 'wydziel zbiory treningowe (treningowy, walidacyjny, testowy)':
        print()
        print('Przygotowanie zbiorów wejściowych w toku, proszę czekać...')
        if not os.path.isfile(preprocessed_data_path):
            print(f"pod ścieżką {preprocessed_data_path} nie został znaleziony plik z wstępnie przetworzonymi danymi, wygeneruj je odpowiednim poleceniem z menu")
            continue
        extract_datasets(settings, preprocessed_data_path, input_sets_path)
        print(f'przetwarzanie zakończone, wynik zapisany pod ścieżką: {input_sets_path}')
        
    elif mm_ans == 'trenuj aproksymator':
        print()
        print('Trening aproksymatora funkcji nagrody.')
        if not os.path.isfile(input_sets_path):
            print(f"pod ścieżką {input_sets_path} nie został znaleziony plik z danymi do treningu, wygeneruj je odpowiednim poleceniem z menu")
            continue
        train_approximator(settings, input_sets_path)
    
    elif mm_ans == 'wyjście':
        break
    
    else:
        raise RuntimeError('Wybrano zły numer akcji w menu głównym.')

