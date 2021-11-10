import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from _imports import *

os.system('cls')

remove_duplicates = ask_for_user_preference('Czy usunąć duplikaty projektów wygenerowanych przez algorytmy?')
verify_designs = ask_for_user_preference('Czy symulacyjnie zweryfikować własności najlepszych projektów?')

# procedury pomocnicze
def show_geometry_preview(settings_sim, pattern, scale_geometries = 3):
    courant_number = settings_sim['basic']['courant_number']
    basic_element_dimensions = settings_sim['diffuser_geometry']['basic_element_dimensions']
    fs = settings_sim['basic']['fs']
    T_air_C        = settings_sim['propagation_medium']['T_air_C']
    p_air_hPa      = settings_sim['propagation_medium']['p_air_hPa']
    RH             = settings_sim['propagation_medium']['RH']
    c, Z_air       = get_air_properties(T_air_C, p_air_hPa, RH)
    T              = 1/fs # [s]
    X              = c*T/courant_number # [m]
    num_element_height_levels = settings_sim['diffuser_geometry']['num_element_height_levels']
    diffuser_depth            = settings_sim['diffuser_geometry']['diffuser_depth']
    shape_skyline = generate_2D_Skyline_diffuser(
                pattern,
                element_seg_depth=cont2disc(diffuser_depth*scale_geometries/num_element_height_levels,X), 
                element_size=cont2disc(basic_element_dimensions*scale_geometries,X))
    show_shape(shape_skyline)

def verify_scattering_properties(settings_sim, pattern, reference_data):
    mean_coeff = evaluate_design(settings_sim, pattern, reference_data)
    print('średnia dyfuzja: ', mean_coeff)
    # print (mean_coeff)
    # draw_subband_polar_response(settings_sim, imp_res_object[0])
    # plt.title('xy')
    # draw_subband_polar_response(settings_sim, imp_res_object[1])
    # plt.title('yz')

def remove_duplicate_designs(patterns, diffusions):
    filtered_patterns   = []
    filtered_diffusions = []
    
    def pattern_in_list(pattern, list):
        list_of_comparisons = []
        for element in list:
            list_of_comparisons.append(np.array_equal(pattern,element))
        return np.any(list_of_comparisons)
    
    already_existing_patterns = []
    for pattern, diffusion in zip(patterns, diffusions):
        if not pattern_in_list(pattern, already_existing_patterns):
            filtered_patterns.append(pattern)
            already_existing_patterns.append(pattern)
            filtered_diffusions.append(diffusion)
    
    return filtered_patterns, filtered_diffusions

# konfiguracja procedur bazujących na AI
CONFIG_PATH_AI          = '_settings/ai_default.ini'
CONFIG_PATH_SIM         = '_settings/sim_default.ini'
settings_ai             = read_config(CONFIG_PATH_AI)
settings_sim            = read_config(CONFIG_PATH_SIM)

algenet_outcomes_dir    = '../_joint_algenet_results'
file_save_dir           = settings_sim['basic']['file_save_dir']
reference_file_path     = os.path.join(file_save_dir,'reference.npy')


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

# odczyt postępu algorytmu genetycznego
algenet_diffusions = []
algenet_patterns   = []
algenet_gen_nums   = []
if os.path.isdir(algenet_outcomes_dir):
    
    for fname in os.listdir(algenet_outcomes_dir):
        _, ext = os.path.splitext(fname)
        if ext != '.npy': continue
        fdata = np.load(os.path.join(algenet_outcomes_dir,fname), allow_pickle=True)
        
        for item in fdata:
            algenet_diffusions.append(item['diffusion'])
            algenet_patterns.append(item['pattern'])
            algenet_gen_nums.append(item['generation_number'])
        
    best_dif_argmax = np.argmax(algenet_diffusions)
    pattern = algenet_patterns[best_dif_argmax]
    dif = algenet_diffusions[best_dif_argmax]
    
    if remove_duplicates:
        algenet_patterns, algenet_diffusions = remove_duplicate_designs(algenet_patterns, algenet_diffusions)
    algenet_best_pattern_idx = np.argmax(algenet_diffusions)

# odczyt danych dla poszukiwania losowego
_, consolidated_data = obtain_replay_folder_contents(settings_ai)

random_diffusions = []
random_patterns   = []
for entry in consolidated_data:
    if 'input_pattern_generation' in list(entry.keys()):
        if entry['input_pattern_generation'] != 'random':
            continue
    
    random_pattern = entry['replay_transitions'][0]['current_pattern']
    random_diffusion = entry['episode_diffusions'][0] - entry['episode_rewards'][0]
    random_diffusions.append(random_diffusion)
    random_patterns.append(random_pattern)

if remove_duplicates:
    random_patterns, random_diffusions = remove_duplicate_designs(random_patterns, random_diffusions)
random_diffusions = np.array(random_diffusions)
random_best_pattern_idx = np.argmax(random_diffusions)

# odczyt danych dla głębokiego gradientu strategii
agent_diffusions_rnd  = []
agent_diffusions_bst  = []
agent_patterns_rnd    = []
agent_patterns_bst    = []
for entry in consolidated_data:
    episode_diffusions_argmax = np.argmax(entry['episode_diffusions'])
    best_pattern   = entry['replay_transitions'][episode_diffusions_argmax]['new_pattern']
    
    if 'input_pattern_generation' in list(entry.keys()):
        if entry['input_pattern_generation'] != 'random':
            agent_diffusions_bst.append(np.max(entry['episode_diffusions']))
            agent_patterns_bst.append(best_pattern)
            continue
    
    agent_diffusions_rnd.append(np.max(entry['episode_diffusions']))
    agent_patterns_rnd.append(best_pattern)

if remove_duplicates:
    agent_patterns_rnd, agent_diffusions_rnd = remove_duplicate_designs(agent_patterns_rnd, agent_diffusions_rnd)
    agent_patterns_bst, agent_diffusions_bst = remove_duplicate_designs(agent_patterns_bst, agent_diffusions_bst)
dpg_best_pattern_bst_idx = np.argmax(agent_diffusions_bst)
dpg_best_pattern_rnd_idx = np.argmax(agent_diffusions_rnd)

print()
print(f'random               - num designs: {len(random_diffusions)}')
print(f'genetic alg.         - num designs: {len(algenet_diffusions)}')
print(f'deep policy gradient (random input)  - num designs: {len(agent_diffusions_rnd)}')
print(f'deep policy gradient (best 10 input) - num designs: {len(agent_diffusions_bst)}')
print()

print()
print(f'best pattern random choice')
print(random_patterns[random_best_pattern_idx])
print(f'provided diffusion: {random_diffusions[random_best_pattern_idx]}')


if os.path.isdir(algenet_outcomes_dir):
    print()
    print(f'best pattern by genetic algorithm (generation no {algenet_gen_nums[algenet_best_pattern_idx]})')
    print(algenet_patterns[algenet_best_pattern_idx])
    print(f'provided diffusion: {algenet_diffusions[algenet_best_pattern_idx]}')

print()
print(f'best pattern by deep policy gradient (random input)')
print(agent_patterns_rnd[dpg_best_pattern_rnd_idx])
print(f'provided diffusion: {agent_diffusions_rnd[dpg_best_pattern_rnd_idx]}')

print()
print(f'best pattern by deep policy gradient (best 10 input)')
print(agent_patterns_bst[dpg_best_pattern_bst_idx])
print(f'provided diffusion: {agent_diffusions_bst[dpg_best_pattern_bst_idx]}')
print()

# Wykreślenie estymat gęstości prawdopodobieństwa
random_diffusions_df = pd.DataFrame()
random_diffusions_df = random_diffusions_df.assign(**{'mean diffusion coefficient':random_diffusions})
random_diffusions_df = random_diffusions_df.assign(**{'algorithm type':'random'})

agent_diffusions_rnd_df = pd.DataFrame()
agent_diffusions_rnd_df = agent_diffusions_rnd_df.assign(**{'mean diffusion coefficient':agent_diffusions_rnd})
agent_diffusions_rnd_df = agent_diffusions_rnd_df.assign(**{'algorithm type':'deep policy gradient (random input)'})

agent_diffusions_bst_df = pd.DataFrame()
agent_diffusions_bst_df = agent_diffusions_bst_df.assign(**{'mean diffusion coefficient':agent_diffusions_bst})
agent_diffusions_bst_df = agent_diffusions_bst_df.assign(**{'algorithm type':'deep policy gradient (best input)'})

algenet_diffusions_df = pd.DataFrame()
if os.path.isdir(algenet_outcomes_dir):
    algenet_diffusions_df = algenet_diffusions_df.assign(**{'mean diffusion coefficient':algenet_diffusions})
    algenet_diffusions_df = algenet_diffusions_df.assign(**{'algorithm type':'genetic algorithm'})

joint_df = pd.concat([random_diffusions_df,agent_diffusions_rnd_df,agent_diffusions_bst_df,algenet_diffusions_df])

print()
print('Mediany rozkładów:')
print(joint_df.groupby('algorithm type').median())
print()
print('Wynik autotestu różnic między dyfuzorami:')
input_df = joint_df.pivot(columns='algorithm type', values='mean diffusion coefficient')
print(input_df)
print(input_df.columns)
autotest(input_df)
print()

sns.histplot(data=joint_df, x='mean diffusion coefficient', hue='algorithm type', multiple="dodge", stat='probability',common_norm=False)
plt.xlabel('diffusion coefficient [-]', fontsize=13)
plt.ylabel('estimated probability [-]', fontsize=13)
plt.xlim([0,1])
plt.grid()
plt.gca().set_yscale('log')

plt.show()


if verify_designs:
    # print("\n----------------------------------------------------------------------")
    # print("weryfikacja własności dyfuzora wytworzonego przez próbkowanie losowe:")
    best_pattern = random_patterns[random_best_pattern_idx]
    verify_scattering_properties(settings_sim, best_pattern, reference_data)
    show_geometry_preview(settings_sim, best_pattern)

    # print("\n----------------------------------------------------------------------")
    print("weryfikacja własności dyfuzora wytworzonego przez algorytm genetyczny:")
    best_pattern = algenet_patterns[algenet_best_pattern_idx]
    verify_scattering_properties(settings_sim, best_pattern, reference_data)
    show_geometry_preview(settings_sim, best_pattern)

    print("\n----------------------------------------------------------------------")
    print("weryfikacja własności dyfuzora wytworzonego przez algorytm głębokiego gradientu strategii:")
    best_pattern = agent_patterns_bst[dpg_best_pattern_bst_idx]
    verify_scattering_properties(settings_sim, best_pattern, reference_data)
    show_geometry_preview(settings_sim, best_pattern)

    plt.show()