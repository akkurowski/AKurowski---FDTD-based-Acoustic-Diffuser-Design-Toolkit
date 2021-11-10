# -------------------------------------------
# Algorytm projektujący dyfuzory metodą
# deep policy gradient
# autor:  Adam Kurowski
# data:   07.02.2021
# e-mail: akkurowski@gmail.com
# -------------------------------------------

import os
import numpy as np
import tensorflow as tf
from _imports import *
import plotly.express as px

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
# Zarządzanie zasobami (GPU)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    specified_running_GPU = settings_ai['reinforcement_learning']['agent_gpu']
    tf.config.experimental.set_visible_devices(gpus[specified_running_GPU], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[specified_running_GPU], True)
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

# -------------------------------------------
# Wykonanie algorytmu projektowania
uni_clear()

print()
print('Utworzenie obiektu agenta uczenia wzmacnianego')
agent = obtain_rl_agent(settings_sim, settings_ai)
load_agent_weights(settings_ai, agent)

# Narysowanie nagłówka programu
print()
print('--------------------------------------------------')
print(' Skrypt uczenia maszynowego poszukujący ')
print(' najlepszych wzorów dyfuzora Schroedera za pomocą')
print(' uczenia wzmacnianego')
print(' autor:         A. Kurowski')
print(' e-mail:        akkurowski@gmail.com')
print(' wersja z dnia: 07.02.2021')
print('--------------------------------------------------')
print()

print('INFORMACJA:')
print(f"\tskrypt wykona się na GPU nr {settings_ai['reinforcement_learning']['agent_gpu']}")
print(f"\tsymulacja wykonywana będzie na GPU nr: {settings_sim['basic']['GPU_device_id']}")
print(f"\tid agenta w skrypcie: {settings_ai['reinforcement_learning']['agent_id']}")
print(f"\tzezwolenie na konsolidację pamięci powtórek: {settings_ai['reinforcement_learning']['allow_replay_consolidation']}")
print()

# Uruchomienie głównej pętli interakcji
while True:
    main_menu_items = []
    main_menu_items.append('załaduj ponownie wagi agenta')
    main_menu_items.append('przeprowadź trening wstępny agenta')
    main_menu_items.append('rozpocznij działanie agenta')
    main_menu_items.append('skonsoliduj pamięć powtórek')
    main_menu_items.append('wyświetl nazwy agentów')
    main_menu_items.append('wyświetl postępy agentów')
    main_menu_items.append('eksportuj postępy agentów')
    main_menu_items.append('zmień nazwy agentów')
    main_menu_items.append('podziel pamięć powtórek na równe części')
    main_menu_items.append('przelicz ponownie wsp. dyfuzji')
    main_menu_items.append('zwizualizuj dane po parametryzacji')
    main_menu_items.append('wyjście')
    
    mm_ans = ask_user_for_an_option_choice('\nWybierz nr akcji do wykonania', 'Nr akcji:', main_menu_items)
    
    if mm_ans == 'załaduj ponownie wagi agenta':
        agent = obtain_rl_agent(settings_sim, settings_ai)
        load_agent_weights(settings_ai, agent)

    elif mm_ans == 'rozpocznij działanie agenta':
        training_type = ask_user_for_an_option_choice('\nWybierz typ treningu', 'Nr typu:', ['losowy projekt początkowy','projekt początkowy z 3. kwartyla','projekt początkowy z 90. percentyla','projekt początkowy wybrany spośród n najlepszych projektów'])
        options_dict = {}
        options_dict.update({'losowy projekt początkowy':'random'})
        options_dict.update({'projekt początkowy z 3. kwartyla':'3rd_quartile'})
        options_dict.update({'projekt początkowy z 90. percentyla':'90th_percentile'})
        options_dict.update({'projekt początkowy wybrany spośród n najlepszych projektów':'best_n'})
        training_type = options_dict[training_type]
        
        kwargs = {}
        if training_type == 'best_n':
            n_best_designs = int(ask_user_for_a_float('\tliczba najlepszych przykładów: '))
            kwargs.update({'n_best_designs':n_best_designs})
        
        try:
            reinforcement_learning_training_loop(settings_sim, settings_ai, reference_data, agent, input_pattern_generation=training_type,**kwargs)
        except KeyboardInterrupt:
            print()
            print()
            print('Algorytm uczenia wzmacnianego został przerwany ręcznie')
    
    elif mm_ans == 'przeprowadź trening wstępny agenta':
        # pretrening modelu
        print()
        num_epochs = ask_user_for_a_float('\tliczba epok treningu na epizod:')
        
        last_episodes_limit = None
        if ask_for_user_preference('\tCzy ograniczyć trening do n ostatnich epok?'):
            last_episodes_limit = int(ask_user_for_a_float('\twpisz liczbę epok:'))
        
        num_cycles=1
        if ask_for_user_preference('\tCzy wykonać kilka cykli uczenia przechodzących przez pamięć powtórek?'):
            num_cycles = int(ask_user_for_a_float('\twpisz liczbę cykli:'))
        
        print()
        print('wstępny pretrening agenta na pamięci powtórek:')
        train_on_replay_memory(settings_ai, agent, int(num_epochs), last_episodes_limit=last_episodes_limit, num_cycles=num_cycles)
        # zapis modelu
        rl_agent_wights_saving_path = os.path.join(settings_ai['reinforcement_learning']['rl_agent_weights_dir'],'rl_agent')
        agent.save_model(rl_agent_wights_saving_path)
    
    elif mm_ans == 'skonsoliduj pamięć powtórek':
        consolidate_replay_memory(settings_ai)
    
    elif mm_ans == 'zmień nazwy agentów':
        rename_agents_in_replay_memory(settings_ai)
    
    elif mm_ans == 'wyświetl nazwy agentów':
        _, consolidated_data = obtain_replay_folder_contents(settings_ai)
        agents_fingerprints = get_replay_agent_fingerprints(consolidated_data)
        print()
        print('Nazwy agentów w pamięci powtórek:')
        for gpu_id, agent_id in agents_fingerprints:
            print(f'\tgpu:{gpu_id}, agent id: {agent_id}')
        print()
    
    elif mm_ans == 'wyświetl postępy agentów':
        _, consolidated_data = obtain_replay_folder_contents(settings_ai)
        
        ans = ask_user_for_an_option_choice('\nJaki typ wizualizacji ma być obliczony', 'Nr opcji:', ['ogólna','pojedynczy agenci na wspólnym wykresie','pojedynczego agenta'])
        
        if ans == 'pojedynczego agenta':
            agents       = get_replay_agent_fingerprints(consolidated_data)
            agents_names = [agent[1] for agent in agents]
            agents_names = sorted(agents_names)
            agent_name   = ask_user_for_an_option_choice('\nWybierz agenta do wizualizacji:', 'Nr agenta:', agents_names)
            
            agent_diffusions, random_diffusions, mean_reward = extract_agents_data(agent_name, consolidated_data)
            
            agent_diffusions  = np.array(agent_diffusions)
            random_diffusions = np.array(random_diffusions)
            
            plt.figure()
            plt.plot(agent_diffusions, label='agent diffusion coeff.')
            plt.plot(mean_reward, label='mean reward')
            plt.xlabel('episode number')
            plt.ylabel('diff. coeff/reward')
            plt.grid()
            plt.legend()
            
            plt.figure()
            plt.hist(agent_diffusions, bins=30, label='agent diffusion coeff.')
            plt.xlim([0,1])
            plt.hist(random_diffusions, bins=30, label='random (initial) diffusion coeff.')
            plt.xlim([0,1])
            plt.grid()
            plt.legend()
            plt.xlabel('diffusion coefficient')
            plt.ylabel('frequency')
            plt.show()
            
        elif ans == 'pojedynczy agenci na wspólnym wykresie':
            agents       = get_replay_agent_fingerprints(consolidated_data)
            agents_names = [agent[1] for agent in agents]
            agents_names = sorted(agents_names)
            
            diff_df   = {}
            reward_df = {}
            for agent_name in agents_names:
                agent_diffusions, _, mean_reward = extract_agents_data(agent_name, consolidated_data)
                diff_df.update({f'{agent_name} diffusion coeff.':agent_diffusions})
                reward_df.update({f'{agent_name} reward':mean_reward})
            
            diff_df = pd.DataFrame.from_dict(diff_df, orient='index').transpose()
            diff_df = pd.melt(diff_df, var_name='agent name', value_name='diffusion coeff.', ignore_index=False)
            
            reward_df = pd.DataFrame.from_dict(reward_df, orient='index').transpose()
            reward_df = pd.melt(reward_df, var_name='agent name', value_name='reward', ignore_index=False)
        
            fig = px.line(diff_df,x=diff_df.index,  y = 'diffusion coeff.', color='agent name')
            fig.update_yaxes(range=[0,1])
            fig.show()
        
            fig = px.line(reward_df,x=reward_df.index,  y = 'reward', color='agent name')
            fig.show()
        
        # kod do refaktoryzacji - else w środku switcha
        # else:
            # random_diffusions = []
            # agent_diffusions  = []
            # for entry in consolidated_data:
                # if 'input_pattern_generation' in list(entry.keys()):
                    # if entry['input_pattern_generation'] == '4th_quartile':
                        # continue
                # agent_diffusions.append(np.max(entry['episode_diffusions']))
                # random_diffusions.append(entry['episode_diffusions'][0])
            # random_diffusions = np.array(random_diffusions)
            
            # plt.figure()
            # plt.hist(agent_diffusions, bins=30, label='agent diffusion coeff.')
            # plt.xlim([0,1])
            # plt.hist(random_diffusions, bins=30, label='random (initial) diffusion coeff.')
            # plt.xlim([0,1])
            # plt.grid()
            # plt.legend()
            # plt.xlabel('diffusion coefficient')
            # plt.ylabel('frequency')
            # plt.gca().set_yscale('log')
            # plt.show()
    
    elif mm_ans == 'zwizualizuj dane po parametryzacji':
            _, consolidated_data = obtain_replay_folder_contents(settings_ai)
            parameterized_example = consolidated_data[54]['replay_transitions'][0]['parameterized_current_pattern']
            pattern         = parameterized_example[:,:,0]
            spectrum        = parameterized_example[:,:,1]
            cepstrum        = parameterized_example[:,:,2]
            autocorrelation = parameterized_example[:,:,3]
            
            plt.matshow(pattern)
            plt.xlabel('segment number (x axis) [-]')
            plt.ylabel('segment number (y axis) [-]')
            plt.colorbar()
            
            plt.matshow(spectrum)
            plt.xlabel('x axis component of spatial spectrum [-]')
            plt.ylabel('y axis component of spatial spectrum [-]')
            plt.colorbar()
            
            plt.matshow(cepstrum)
            plt.xlabel('x axis component of spatial cepstrum [-]')
            plt.ylabel('y axis component of spatial cepstrum [-]')
            plt.colorbar()
            
            plt.matshow(autocorrelation)
            plt.xlabel('x axis component of pattern autocorrelation [-]')
            plt.ylabel('y axis component of pattern autocorrelation [-]')
            plt.colorbar()
            
            plt.show()
            
            exit()
    
    elif mm_ans == 'eksportuj postępy agentów':
            _, consolidated_data = obtain_replay_folder_contents(settings_ai)
            random_diffusions = []
            agent_diffusions  = []
            for entry in consolidated_data:
                agent_diffusions.append(np.max(entry['episode_diffusions']))
                random_diffusions.append(entry['episode_diffusions'][0].keys())
            random_diffusions = np.array(random_diffusions)
            
            output={}
            output.update({'random_diffusions':random_diffusions})
            output.update({'agent_diffusions':agent_diffusions})
            np.save('agents_progress_export.npy',output)
    
    elif mm_ans == 'podziel pamięć powtórek na równe części':
            _, consolidated_data = obtain_replay_folder_contents(settings_ai)
            n_files = int(ask_user_for_a_float('\tliczba plików: '))
            
            file_contents = []
            for i in range(n_files):
                file_contents.append([])
            
            for i in range(len(consolidated_data)):
                part_idx = i%n_files
                file_contents[part_idx].append(consolidated_data[i])
            
            for i in range(n_files):
                np.save(f'segmented_replay_memory_{create_timestamp()}_{str(i+1).zfill(3)}.npy',file_contents[i])
    
    elif mm_ans == 'przelicz ponownie wsp. dyfuzji':
        file_path = input('podaj ścieżkę pliku z danymi: ')
        data_to_process = np.load(file_path, allow_pickle=True)
        print(data_to_process[0]['episode_best_design'])
        exit()
    
    elif mm_ans == 'wyjście':
        break
    
    else:
        raise RuntimeError('Wybrano zły numer akcji w menu głównym.')
