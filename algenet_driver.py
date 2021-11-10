# -------------------------------------------
# Algorytm projektujący dyfuzory metodą
# algorytmu genetycznego
# autor:  Adam Kurowski
# data:   08.02.2021
# e-mail: akkurowski@gmail.com
# -------------------------------------------

import os
import numpy as np
import tensorflow as tf
from _imports import *

# -------------------------------------------
# Odczyt pliku konfiguracyjnego i inicjalizacja

# wyczyszczenie ekranu
def uni_clear():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')
uni_clear()

# konfiguracja procedur bazujących na AI
CONFIG_PATH_AI  = '_settings/ai_default.ini'
settings_ai     = read_config(CONFIG_PATH_AI)

# konfiguracja procedur symulacyjnych
CONFIG_PATH_SIM = '_settings/sim_default.ini'
settings_sim    = read_config(CONFIG_PATH_SIM)

# wydzielenie ścieżek plików z danymi referencyjnymi
# (do oblicznaia charakterystyk dyfuzorów)
file_save_dir       = settings_sim['basic']['file_save_dir']
reference_file_path = os.path.join(file_save_dir,'reference.npy')

# odczyt danych referencyjnych do pomiaru dyfuzora
try:
    print('obliczanie danych referencyjnych:')
    reference_data = np.load(reference_file_path, allow_pickle=True).item()
except:
    print(f'odczyt plik z danymi referencyjnymi ({reference_file_path}) nie powiódł się, rreferencja zostanie obliczona automatycznie')
    imp_res_set_empty, imp_res_set_plate, _ = run_simulation_for_pattern(None,settings_sim, mode='reference_only')
    
    reference_data = {
        'plate':imp_res_set_plate,
        'room':imp_res_set_empty,
        'num_element_height_levels':settings_sim['diffuser_geometry']['num_element_height_levels'],
        'diffuser_depth':settings_sim['diffuser_geometry']['diffuser_depth'],
        'basic_element_dimensions':settings_sim['diffuser_geometry']['basic_element_dimensions'],
        'fs':settings_sim['basic']['fs']}
    
    # Zapis wyników obliczeń na dysk.
    np.save(reference_file_path,reference_data)

# -------------------------------------------
# procedury pomocnicze

# pomocnicza procedura służąca do podejmowania losowych decyzji w 
# procesie optymalizacji genetycznej
def decide_on_random_action(sub_prob):
    perform_substitution = True
    if np.random.uniform() > sub_prob:
        perform_substitution = False
    return perform_substitution

# crossing-over - definicja procedury
def cv_procedure(design_a, design_b, sub_prob = 0.2):
    dim_x      = design_a.shape[0]
    dim_y      = design_a.shape[1]
    new_design = np.zeros_like(design_a)
    
    substitute_rows = True
    if np.random.randint(0,1) == 0:
        substitute_rows = False
    
    if substitute_rows:
        for row_i in range(dim_y):
            if decide_on_random_action(sub_prob):
                new_design[:,row_i] = design_b[:,row_i]
            else:
                new_design[:,row_i] = design_a[:,row_i]
    else:
        # substitute cols
        for col_i in range(dim_x):
            if decide_on_random_action(sub_prob):
                new_design[col_i,:] = design_b[col_i,:]
            else:
                new_design[col_i,:] = design_a[col_i,:]
    
    return new_design

# mutacje - definicja procedury
def mutate_design(settings_sim, pattern, partial_mut_prob=0.2):
    old_pattern = cp.copy(pattern)
    for i in range(pattern.shape[0]):
        for j in range(pattern.shape[1]):
            # czy dla koordynatów i,j wykonać mutację?
            if decide_on_random_action(partial_mut_prob):
                # jeśli tak, to dodać czy odjąć 1 od
                # wartości elementu?
                change_val = 1
                if decide_on_random_action(0.5):
                    change_val = -1
                pattern[i,j] += change_val
    pattern[pattern>settings_sim['diffuser_geometry']['num_element_height_levels']] = 10
    pattern[pattern<0]  = 0
    return pattern

def generate_initial_ga_population(settings_sim, population_size):
    diffuser_dimensions = settings_sim['diffuser_geometry']['diffuser_pattern_size']
    # wygenerowanie timestampu aktualnej instancji skryptu
    run_timestamp = 'algenet_run_'+create_timestamp()
    
    # utworzenie folderu, w którym przechowywane będą dane obecnego 
    # uruchomienia algorytmu genetycznego
    run_path = os.path.join('..',run_timestamp)
    if not os.path.isdir(run_path):
        os.mkdir(run_path)
    
    # Wygenerowanie i ocena nowej populacji osobników
    designs_population = []
    
    for i in range(population_size):
        print(f"\ngenerowanie osobnika {i+1}/{population_size}")
        new_pattern, _    = generate_pattern(settings_sim, diffuser_dimensions)
        pattern_diffusion = evaluate_design(settings_sim, new_pattern, reference_data)
        descriptor = {}
        descriptor.update({'pattern':  new_pattern})
        descriptor.update({'diffusion':pattern_diffusion})
        descriptor.update({'generation_number':1})
        print()
        print(new_pattern)
        print()
        print(f"obliczona dyfuzja: {pattern_diffusion}")
        print()
        designs_population.append(descriptor)
    
    return designs_population, run_path

def run_genetic_algorithm(designs_population, run_path, max_best, population_size, init_generation_id=1):
    print()
    run_timestamp = 'algenet_run_'+create_timestamp()
    # nieskończona pętla algorytmu genetycznego
    generation_id = init_generation_id
    while True:
        print(f'pokolenie nr {generation_id}')
        # posortowanie zbioru dyfuzorów i ocen tak aby powstała lista
        # rankingowa
        designs_population = sorted(designs_population, key=lambda k: k['diffusion'], reverse=True) 
        if len(designs_population) > population_size:
            designs_population = designs_population[0:population_size]
        
        # zapis wyniku do folderu
        save_path = os.path.join(run_path,f"gen_{str(generation_id).zfill(5)}_algenet_{run_timestamp}.npy")
        np.save(save_path, designs_population)
        
        # uaktualniamy id generacji algorytmu genetycznego
        generation_id += 1
        
        # crossing-over i mutacja
        new_designs_population = []
        best_designs = designs_population[0:max_best]
        new_design_id = 1
        print()
        
        for i, desc_design_a in enumerate(best_designs):
            for j, desc_design_b in enumerate(best_designs):
                if i == j: continue # nie łączymy ze sobą tych samych projektów
                
                design_a = desc_design_a['pattern']
                design_b = desc_design_b['pattern']
                
                print(f"generowanie projektu pochodnego nr {new_design_id}")
                
                new_design = cv_procedure(design_a, design_b)
                new_design = mutate_design(settings_sim, new_design)
                pattern_diffusion = evaluate_design(settings_sim, new_design, reference_data)
                print()
                print(new_design)
                print()
                print(f"obliczona dyfuzja: {pattern_diffusion}")
                print()
                
                descriptor = {}
                descriptor.update({'pattern':  new_design})
                descriptor.update({'diffusion':pattern_diffusion})
                descriptor.update({'generation_number':generation_id})
                
                new_designs_population.append(descriptor)
                new_design_id += 1
        
        new_designs_population += best_designs
        print()
        designs_population = new_designs_population

# -------------------------------------------
# Wykonanie algorytmu projektowania

# Narysowanie nagłówka programu
print()
print('--------------------------------------------------')
print(' Skrypt uczenia maszynowego poszukujący ')
print(' najlepszych wzorów dyfuzora Schroedera za pomocą')
print(' algorytmu genetycznego')
print(' autor:         A. Kurowski')
print(' e-mail:        akkurowski@gmail.com')
print(' wersja z dnia: 08.02.2021')
print('--------------------------------------------------')
print()

print('INFORMACJA:')
print(f"\tsymulacja wykonywana będzie na GPU nr: {settings_sim['basic']['GPU_device_id']}")
print()

max_best           = 6
population_size    = max_best*(max_best-1)

# Uruchomienie głównej pętli interakcji
while True:
    main_menu_items = []
    main_menu_items.append('rozpocznij działanie algorytmu genetycznego')
    main_menu_items.append('wznów działanie algorytmu genetycznego')
    main_menu_items.append('wyjście')
    
    mm_ans = ask_user_for_an_option_choice('\nWybierz nr akcji do wykonania', 'Nr akcji:', main_menu_items)
    
    if mm_ans == 'rozpocznij działanie algorytmu genetycznego':
        designs_population, run_path = generate_initial_ga_population(settings_sim, population_size)
        try:
            run_genetic_algorithm(designs_population, run_path, max_best, population_size)
        except:
            print('wykonanie algorytmu przerwane ręcznie')
    
    if mm_ans == 'wznów działanie algorytmu genetycznego':
        data_dir = input('podaj ścieżkę do folderu z danymi: ')
        
        latest_generation_fdata = None
        for data_fname in os.listdir(data_dir):
            dfile_fpath = os.path.join(data_dir, data_fname)
            try:
                fdata = np.load(dfile_fpath,allow_pickle=True)
            except:
                print(f"błąd odczytu pliku: {dfile_fpath}")
                continue
            
            if latest_generation_fdata is None:
                latest_generation_fdata = fdata
            else:
                if latest_generation_fdata[0]['generation_number'] < fdata[0]['generation_number']:
                    latest_generation_fdata = fdata
        
        if latest_generation_fdata is not None:
            init_generation_id = latest_generation_fdata[0]['generation_number']
            try:
                run_genetic_algorithm(latest_generation_fdata, data_dir,max_best, population_size, init_generation_id=init_generation_id)
            except KeyboardInterrupt:
                print('wykonanie algorytmu przerwane ręcznie')
        else:
            print('Nie udało wczytać się żadnego pliku z danymi')
        
    elif mm_ans == 'wyjście':
        break
    
    else:
        raise RuntimeError('Wybrano zły numer akcji w menu głównym.')
