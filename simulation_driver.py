# ---------------------------------------------------- #
# Main simulation package used for predicting impulse
# responses of given Schroeder diffuser designs
# autor: Adam Kurowski
# data:   24.12.2020
# e-mail: akkurowski@gmail.com
# ---------------------------------------------------- #

from _imports import *
import numpy as np
import shutil
import os

if os.name == 'nt':
    os.system('cls')
else:
    os.system('clear')

# -------------------------------------------------------
# Odczyt nastaw z pliku konfiguracyjnego
# -------------------------------------------------------

# Odczyt pliku konfiguracyjnego.
CONFIG_PATH = '_settings/sim_default.ini'
settings    = read_config(CONFIG_PATH)

# -------------------------------------------------------
# Wykonanie pliku
# -------------------------------------------------------

# Zablokowanie generatora losowego
if settings['basic']['lock_random_generator']:
    np.random.seed(10012021)

# Odczyt zapisanych charakterystyk (jeśli są zachowane)
imp_res_set_empty   = None
imp_res_set_plate   = None
imp_res_object      = None

file_save_dir       = settings['basic']['file_save_dir']
reference_file_path = os.path.join(file_save_dir,'reference.npy')
diffuser_file_path  = os.path.join(file_save_dir,'diffuser.npy')

try:
    if os.path.isfile(reference_file_path):
        ref_data = np.load(reference_file_path, allow_pickle=True).item()
        imp_res_set_empty = ref_data['room']
        imp_res_set_plate = ref_data['plate']
except:
    print(f'odczyt plik z danymi referencyjnymi ({reference_file_path}) nie powiódł się')

try:
    if os.path.isfile(diffuser_file_path):
        ref_data = np.load(diffuser_file_path, allow_pickle=True).item()
        imp_res_object = ref_data['object']
except:
    print(f'odczyt plik z danymi referencyjnymi ({diffuser_file_path}) nie powiódł się')

# Narysowanie nagłówka programu
print()
print('--------------------------------------------------')
print(' Symulator zachowania się dyfuzorów Skyline')
print(' metodą FDTD')
print(' autor:         A. Kurowski')
print(' e-mail:        akkurowski@gmail.com')
print(' wersja z dnia: 28.01.2021')
print('--------------------------------------------------')
print()

# Uruchomienie głównej pętli interakcji
while True:
    main_menu_items = []
    main_menu_items.append('oblicz charakterystyki referencyjne (pomieszczenie, płyta)')
    main_menu_items.append('oblicz charakterystykę dyfuzora')
    main_menu_items.append('pokaż charakterystyki kierunkowe dyfuzora')
    main_menu_items.append('wyjście')
    
    mm_ans = ask_user_for_an_option_choice('\nWybierz nr akcji do wykonania', 'Nr akcji:', main_menu_items)
    
    if mm_ans == 'oblicz charakterystyki referencyjne (pomieszczenie, płyta)':
        # Obliczenie charakterystyk referencyjnych.
        imp_res_set_empty, imp_res_set_plate, _ = run_simulation_for_pattern(None,settings, mode='reference_only')
        
        # Zapis wyników obliczeń na dysk.
        np.save(reference_file_path,{
            'plate':imp_res_set_plate,
            'room':imp_res_set_empty,
            'num_element_height_levels':settings['diffuser_geometry']['num_element_height_levels'],
            'diffuser_depth':settings['diffuser_geometry']['diffuser_depth'],
            'basic_element_dimensions':settings['diffuser_geometry']['basic_element_dimensions'],
            'fs':settings['basic']['fs']})
        
        print('\nObliczenia referencyjne zostały wykonane, a wynik zapisano do folderu _stored_data\n')

    elif mm_ans == 'oblicz charakterystykę dyfuzora':
        
        while True:
            try:
                # Zaprojektowanie patternu.
                x_nb = settings['diffuser_geometry']['diffuser_pattern_size'][1]
                y_nb = settings['diffuser_geometry']['diffuser_pattern_size'][0]
                pattern = np.random.randint(0,settings['diffuser_geometry']['num_element_height_levels'],(x_nb,y_nb))
        
                # Symulacja zachowania się geometrii dyfuzora.
                _, _, imp_res_object = run_simulation_for_pattern(pattern,settings, mode='shape_only')
                
                # Zapis wyników obliczeń na dysk.
                np.save(diffuser_file_path,{
                    'pattern':pattern,
                    'object':imp_res_object,
                    'num_element_height_levels':settings['diffuser_geometry']['num_element_height_levels'],
                    'diffuser_depth':settings['diffuser_geometry']['diffuser_depth'],
                    'basic_element_dimensions':settings['diffuser_geometry']['basic_element_dimensions'],
                    'fs':settings['basic']['fs']})
                
                if settings['basic']['archivize_diffuser_results']:
                    arch_fname = os.path.join(settings['basic']['file_save_dir'],f'diffuser_{create_timestamp()}.npy')
                    shutil.copyfile(diffuser_file_path, arch_fname)
                
                print('\nObliczenia zachowania się dyfuzora zostały wykonane, a wynik zapisano do folderu _stored_data\n')
                if not settings['basic']['infinite_diffuser_testing']:
                    break
            except KeyboardInterrupt as e:
                print('\nObliczenia testujące geometrie dyfuzora zostały przerwane ręcznie.')
                break
    
    elif mm_ans == 'pokaż charakterystyki kierunkowe dyfuzora':
    
        if (imp_res_set_empty is None) or (imp_res_set_plate is None):
            print('\nNie zostały obliczone ani zachowane dane referencyjne - oblicz je za pomocą odpowiedniej opcji w menu.')
            continue
            
        if imp_res_object is None:
            print('\nNie został jeszcze obliczony wynik dla badanego dyfuzora - oblicz go za pomocą odpowiedniej opcji w menu.')
            continue
        
        polar_resp_plt_xy = wideband_polar_response(imp_res_set_plate[0]-imp_res_set_empty[0])
        polar_resp_obj_xy = wideband_polar_response(imp_res_object[0]-imp_res_set_empty[0])
        polar_resp_plt_yz = wideband_polar_response(imp_res_set_plate[1]-imp_res_set_empty[1])
        polar_resp_obj_yz = wideband_polar_response(imp_res_object[1]-imp_res_set_empty[1])
        
        draw_polar_response([polar_resp_plt_xy, polar_resp_obj_xy],labels=['xy plate','xy skyline'])
        plt.title('charakterystyka kierunkowa w pł. XY')
        
        draw_polar_response([polar_resp_plt_yz,polar_resp_obj_yz],labels=['yz plate','yz skyline'])
        plt.title('charakterystyka kierunkowa w pł. YZ')
        
        plt.show()
    
    elif mm_ans == 'wyjście':
        break
    
    else:
        raise RuntimeError('Wybrano zły numer akcji w menu głównym.')

