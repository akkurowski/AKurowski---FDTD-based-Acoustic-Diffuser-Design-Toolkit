# -------------------------------------------
# Procedury realizujące algorytmy uczenia
# wzmacnianego
# autor:  Adam Kurowski
# data:   07.02.2021
# e-mail: akkurowski@gmail.com
# -------------------------------------------
import time
import numpy as np
import os
import copy as cp
import shutil

from _imports import *

import tensorflow as tf
from tensorflow.keras.optimizers  import Adam
from tensorflow.keras.layers      import Input, Dense, Conv2D, Flatten, Reshape, Lambda, PReLU, BatchNormalization
from tensorflow.keras.models      import Model
from tensorflow.keras.activations import softmax
from tensorflow.keras.backend     import sum
from tensorflow.keras import backend as K

# -------------------------------------------
# Definicje modeli uczenia maszynowego

class DPGAgent(object):
    # Inicjalizacja obiektu zarządzającego agentem RL i procesem
    # treningu.
    def __init__(self, input_dims, num_diff_elem_hlevels, lr=0.001):
        
        # liczba poziomów wysokości dyfuzora
        self.num_diff_elem_hlevels = num_diff_elem_hlevels
        
        # nastawy algorytmu uczenia wzmacnianego
        # wsp. dyskontowania
        self.gamma         = 0.99
        # zakumulowana nagroda po zdyskontowaniu odległych 
        # nagród cząstkowych
        self.G             = 0
        
        # zmienne przejść stanów
        self.state_memory  = []
        self.action_memory = []
        self.reward_memory = []
        
        # wymiary wejściowe sieci
        self.input_dims    = input_dims
        
        # nastawy i obiekt optymalizatora
        self.lr            = lr
        self.optimizer     = Adam(lr=self.lr)
        
        # struktura sieci DPG
        self.policy, self.predict = self.build_policy_network(input_dims,lr)
    
    # Definicja struktury sieci agenta RL
    def build_policy_network(self, input_dims, lr):
        
        input      = Input(shape=(*input_dims,4))
        advantages = Input(shape=(1))\
        
        def resnet_block(input, hdn_layers = 1, num_chnls=16, mask=(2,2), bnorm_momentum=0.95):
            x  = BatchNormalization(momentum = bnorm_momentum)(input)
            xA = Conv2D(num_chnls,mask,  activation=PReLU(), padding='same')(x)
            x  = xA
            x  = BatchNormalization(momentum = bnorm_momentum)(x)
            for i in range(hdn_layers):
                x = Conv2D(num_chnls,mask,  activation=PReLU(), padding='same')(x)
                x = BatchNormalization(momentum = bnorm_momentum)(x)
            x = Add()([x,xA])
            return x
        
        x  = resnet_block(input, hdn_layers = 4, num_chnls=16, mask=(4,4))
        x  = resnet_block(x,     hdn_layers = 4, num_chnls=32, mask=(3,3))
        xA = resnet_block(x,     hdn_layers = 4, num_chnls=64, mask=(2,2))
        x  = resnet_block(xA,    hdn_layers = 4, num_chnls=64, mask=(2,2))
        x  = resnet_block(x,     hdn_layers = 4, num_chnls=64, mask=(2,2))
        x  = Add()([x,xA])
        x  = Conv2D(32,(2,2), activation=PReLU(), padding='same')(x)
        x  = Conv2D(16,(2,2), activation=PReLU(), padding='same')(x)
        x  = Conv2D(8,(2,2),  activation=PReLU(), padding='same')(x)
        x  = Conv2D(3,(2,2),  activation=PReLU(), padding='same')(x)
        probs = Lambda(lambda x: K.softmax(x, axis=-1))(x)
        
        # model do treningu sieci neuronowej DPG
        policy  = Model(inputs=[input, advantages], outputs=[probs])
        # model do inferencji
        predict = Model(inputs=[input], outputs=[probs])
        
        return (policy, predict)
    
    # wybór akcji przez agenta
    def choose_action(self, observation):
        # konwersja macierzy prawdopodobnieństw decyzji wyliczonych
        # przez sieć do formatu prawdopodobieństwo -> akcja
        
        #   dodanie osi wymaganych przez TensorFlow
        observation  = observation[np.newaxis,:,:,:]
        
        #   inferencja
        probs_matrix = self.predict.predict(observation)
        
        #   podjęcie decyzji na podstawie probabilistycznej odpowiedzi z sieci
        actions = np.zeros((probs_matrix.shape[1],probs_matrix.shape[2]))
        
        # pętla realizująca wybór akcji na podstawie wyjścia sieci
        for i in range(actions.shape[0]):
            for j in range(actions.shape[1]):
                probs        = probs_matrix[0,i,j,:]
                actions[i,j] = np.random.choice([-1,0,1], p=probs)
        
        actions = actions.astype(np.int)
        
        # przygotowanie struktury akcji i zwrócenie jej przez funkcję
        action_dict = {}
        action_dict.update({'probs_matrix':probs_matrix})
        action_dict.update({'actions_mtx':actions})
        
        return action_dict
    
    def store_transition(self, observation, action, reward):
        self.state_memory .append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
    
    def learn(self,num_epochs=1):
        # Konnwersja list do formatu Numpy
        state_memory    = np.array(self.state_memory)
        action_memory   = np.array(self.action_memory)
        reward_memory   = np.array(self.reward_memory)
        
        # konwersja wszystkich akcji w pamięci agenta na 
        # postać macierzową
        actions = []
        for i in range(len(action_memory)):
            actions.append(action_memory[i])
        actions = np.array(actions)
        
        # Obliczenie historii wartości zdyskontowanej 
        # nagrody sumarycznej (G). Historia zmian jest potrzebna
        # w funkcji straty (mnoży się przez nią kolejne nagrody)
        G = np.zeros_like(reward_memory)
        # pętla obliczająca
        for t in range(len(reward_memory)):
            # inicjalizacja dyskontowania i obliczania sumarycznej 
            # wartości G
            G_sum         = 0
            discount      = 1
            
            # pętla obliczająca
            for k in range(t, len(reward_memory)):
                # dodajemy kolejne zdyskontowane wartości
                G_sum    += reward_memory[k]*discount
                # aktualna wartyość przeceny jest uaktualniany
                # za pomocą mnożenie przez współczynnik 
                # dyskontowania
                discount *= self.gamma
            
            # kolejne kroki nagrody są uaktualniane 
            G[t] = G_sum
        
        # standardyzacja zakumulowanych wartości zdyskontowanych nagród
        # zabezpieczenie przed zerowym odchyleniem standardowym i 
        # dodanie osi na potrzeby dalszych procedur treningu
        mean         = np.mean(G)
        std          = np.std(G) if np.std(G) > 0 else 1
        self.G       = (G-mean)/std

        # funkcja straty bazująca na historii zdyskontowanych nagród i odpowiedziach sieci
        def custom_loss(model, y_true, y_pred, discounted_rewards):
            # clipping wyjścia sieci - dla zabezpiecznia zbieżności
            out     = tf.clip_by_value(y_pred, 1e-8, 1-1e-8)
            
            # obliczenie krosentropii
            log_lik = y_true*tf.math.log(out)
            t       = sum(-log_lik,axis=3)
            t       = sum(t,axis=3)
            t       = sum(t,axis=2)
            t       = sum(t,axis=1)
            
            # obliczenie straty (ważenie zakumulowanym 
            # i zdyskontowanym sygnałem nagrody)
            loss_value = sum(t*discounted_rewards)
            
            return loss_value, tape.gradient(loss_value, model.trainable_variables)
        
        # pętla treningu
        for epoch_i in range(num_epochs):
            with tf.GradientTape() as tape:
                # pobranie odpowiedzi sieci na zadane stany
                y_pred = self.predict(tf.convert_to_tensor(state_memory, dtype='float32'))
                
                # obliczenie funkcji straty
                loss_values, grads = custom_loss(
                    self.policy,
                    tf.convert_to_tensor(actions, dtype='float32'),
                    y_pred,
                    tf.convert_to_tensor(self.G, dtype='float32'))
                
                # aplikacja gradientów i wyświetlenie cząstkowego komunikatu i stanie treningu
                self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
                print('epoch:', epoch_i+1, 'loss:', loss_values.numpy(), end='\r')
        print()
        # wyzerowanie pamięci przejść stanów agenta
        self.state_memory  = []
        self.action_memory = []
        self.reward_memory = []
    
    # Procedura zapisu modelu na dysk
    def save_model(self, model_file):
        self.policy.save_weights(model_file)
    
    # Procedura odczytu modelu z dysku
    def load_model(self, model_file):
        self.policy.load_weights(model_file)

# generator patternów
def generate_pattern(settings_sim, diffuser_dimensions):
    pattern = np.random.randint(0,10,size=diffuser_dimensions)
    parameterized_pattern = single_shot_schroeder_parameterize(pattern/settings_sim['diffuser_geometry']['num_element_height_levels'])
    return pattern, parameterized_pattern

# Wrapper funkcji oceniającej wygenerowane przez DPG
# wzory dyfuzorów

def evaluate_design(settings_sim, design_pattern, reference_data):
    print('\npostęp symulacji zachowania się dyfuzora:')
    _, _, imp_res_object   = run_simulation_for_pattern(design_pattern,settings_sim, mode='shape_only')
    print()
    
    shape_data = {
        'pattern':design_pattern,
        'object':imp_res_object,
        'num_element_height_levels':settings_sim['diffuser_geometry']['num_element_height_levels'],
        'diffuser_depth':settings_sim['diffuser_geometry']['diffuser_depth'],
        'basic_element_dimensions':settings_sim['diffuser_geometry']['basic_element_dimensions'],
        'fs':settings_sim['basic']['fs']}
    
    _, design_quality = wideband_diffusion_coefficients(reference_data, shape_data)
    
    return design_quality

def train_on_replay_memory(settings_ai, agent, num_epochs=1, last_episodes_limit = None, num_cycles=1):
    replay_memory_dir = settings_ai['reinforcement_learning']['replay_memory_dir']
    
    # Ładowanie danych o przejściach stanów
    episodes_data_list = []
    
    # Dla każdego pliku z folderu pamięci powtórek:
    if not os.path.isdir(replay_memory_dir):  
        os.mkdir(replay_memory_dir)
    
    for file in os.listdir(replay_memory_dir):
        # odczyt danych o przejściu stanów
        repmem_file_path = os.path.join(replay_memory_dir,file)
        transition_array = np.load(repmem_file_path, allow_pickle=True)
        for i in range(len(transition_array)):
            episodes_data_list.append(transition_array[i])

    # Posiadamy pamięć powtórek - ładujemy jej zawartość do
    # pamięci agenta i trenujemy go co MAX_ITER powtórek
    # Na początek przetrenowujemy algorytm na podstawie zgromadzonej
    # do tej pory zawartości pamięci powtórek.
    if len(episodes_data_list) != 0:
        print()
        print('pretrening agenta na danych powtórek:')
        print()
    
    if last_episodes_limit is not None:
        episodes_data_list = sorted(episodes_data_list, key=lambda x:x['timestamp'])
        last_idx = len(episodes_data_list)-1
        episodes_data_list = episodes_data_list[last_idx-last_episodes_limit:last_idx]
    
    for cycle_i in range(num_cycles):
        print(f'numer cyklu: {cycle_i+1}')
        for i, episode_data in enumerate(episodes_data_list):
            print(f"trening dla pliku {i+1}/{len(episodes_data_list)}")
            for transition in episode_data['replay_transitions']:
                observation     = transition['parameterized_current_pattern']
                prob_action     = transition['action_dict']['probs_matrix']
                reward          = transition['reward']
                agent.store_transition(observation, prob_action, reward)
            agent.learn(num_epochs)

def obtain_replay_folder_contents(settings_ai):
    replay_memory_dir   = settings_ai['reinforcement_learning']['replay_memory_dir']
    
    consolidated_data   = []
    detected_data_files = []
    for fname in os.listdir(replay_memory_dir):
    
        fpath  = os.path.join(replay_memory_dir, fname)
        _, ext = os.path.splitext(fname)
        
        if ext == '.npy':
            data_array = np.load(fpath, allow_pickle=True)
            consolidated_data.append(data_array)
            detected_data_files.append(fpath)
    
    if len(consolidated_data) > 0:
        consolidated_data = np.concatenate(consolidated_data,axis=0)
    
    return detected_data_files, consolidated_data

def consolidate_replay_memory(settings_ai):
    replay_memory_dir   = settings_ai['reinforcement_learning']['replay_memory_dir']
    detected_data_files, consolidated_data = obtain_replay_folder_contents(settings_ai)
    
    already_existing_timestamps = []
    filtered_consolidated_data  = []
    for entry in consolidated_data:
        if entry['timestamp'] not in already_existing_timestamps:
            filtered_consolidated_data.append(entry)
            already_existing_timestamps.append(entry['timestamp'])
    
    if len(filtered_consolidated_data) > 0:
        save_path = os.path.join(replay_memory_dir,f"consolidated_{create_timestamp()}")
        np.save(save_path,filtered_consolidated_data)
        
        for fpath in detected_data_files:
            os.remove(fpath)
    
    print('konsolidacja pamięci powtórek przeprowadzona pomyślnie')

def get_replay_gpus(consolidated_data):
    gpu_ids = []
    for fdata in consolidated_data:
        gpu_ids.append(fdata['agent_gpu'])
    gpu_ids = set(gpu_ids)
    return gpu_ids

def get_replay_agent_fingerprints(consolidated_data):
    gpu_ids = get_replay_gpus(consolidated_data)
    agents_fingerprints = []
    for gpu_id in gpu_ids:
        for fdata in consolidated_data:
            if fdata['agent_gpu'] == gpu_id:
                agents_fingerprints.append((gpu_id, fdata['agent_id']))
    agents_fingerprints = set(agents_fingerprints)
    return agents_fingerprints

def rename_agents_in_replay_memory(settings_ai):
    detected_data_files, consolidated_data = obtain_replay_folder_contents(settings_ai)
    gpu_ids = get_replay_gpus(consolidated_data)
    agents_fingerprints = get_replay_agent_fingerprints(consolidated_data)
    
    renaming_dict = {}
    for gpu_id in gpu_ids:
        renaming_dict.update({gpu_id:{}})
    
    for agent_fingerprint in agents_fingerprints:   
        print(f'\ndotychczasowa nazwa agenta: {agent_fingerprint[1]}, gpu: {agent_fingerprint[0]}')
        new_agent_id = input('nowa nazwa agenta (pusta by pominąć):')
        if new_agent_id == '':
            new_agent_id = None
        renaming_dict[agent_fingerprint[0]].update({agent_fingerprint[1]:new_agent_id})
    print()
    
    for i, fdata in enumerate(consolidated_data):
        entry_gpu = fdata['agent_gpu']
        old_id    = fdata['agent_id']
        new_name  = renaming_dict[entry_gpu][old_id]
        
        if new_name is not None:
            consolidated_data[i]['agent_id'] = new_name
    
    replay_memory_dir   = settings_ai['reinforcement_learning']['replay_memory_dir']
    save_path = os.path.join(replay_memory_dir,f"renamed_{create_timestamp()}")
    np.save(save_path,consolidated_data)
    print('zmiana nazw agentów została przeprowadzona pomyślnie')

def load_agent_weights(settings_ai, agent):
    rl_agent_weights_fpath = os.path.join(
        settings_ai['reinforcement_learning']['rl_agent_weights_dir'],
        'rl_agent')
        
    if os.path.isfile(rl_agent_weights_fpath):
        try:
            agent.load_model(rl_agent_weights_fpath)
            print('wagi sieci agenta załadowane pomyślnie')
        except:
            print(f'nie udało się załadować wag sieci agenta, ścieżka ładowania: {rl_agent_weights_fpath}')

# procedura wygenerowania i pretrenowania agenta uczenia wzmacnianego
def obtain_rl_agent(settings_sim, settings_ai, lr=0.0001):
    MAX_ITER            = settings_ai['reinforcement_learning']['episode_length']
    diffuser_dimensions = settings_sim['diffuser_geometry']['diffuser_pattern_size']
    agent               = DPGAgent(diffuser_dimensions, num_diff_elem_hlevels = settings_sim['diffuser_geometry']['num_element_height_levels'], lr=lr)
    
    return agent

def extract_agents_data(agent_name, consolidated_data):
    agent_diffusions  = []
    random_diffusions = []
    mean_reward       = []
    for entry in consolidated_data:
        if entry['agent_id'] == agent_name:
            agent_diffusions.append(np.max(entry['episode_diffusions']))
            random_diffusions.append(entry['episode_diffusions'][0])
            mean_reward.append(np.mean(entry['episode_rewards']))
    return agent_diffusions, random_diffusions, mean_reward

# Główna pętla ucząca
def reinforcement_learning_training_loop(settings_sim, settings_ai, reference_data, agent, input_pattern_generation = 'random', n_best_designs=100):
    
    replay_memory_dir   = settings_ai['reinforcement_learning']['replay_memory_dir']
    diffuser_dimensions = settings_sim['diffuser_geometry']['diffuser_pattern_size']
    MAX_ITER            = settings_ai['reinforcement_learning']['episode_length']
    
    # Inicjalizacja zmiennych do śledzenia postępów treningu
    episode_number = 1
    score_history  = []
    
    while True:
        print()
        print('......................')
        print('epizod nr: %i'%episode_number)
        
        # Wygenerowanie wejściowego projektu dyfuzora (losowego)
        if input_pattern_generation == 'random':
            current_pattern, parameterized_current_pattern = generate_pattern(settings_sim, diffuser_dimensions)
            # Obliczenie jakości wejściowego projektu dyfuzora
            # Symulacja zachowania się geometrii dyfuzora.
            old_diffusion = evaluate_design(settings_sim, current_pattern, reference_data)
        
        # Możemy też wybrać już istniejący projekt, który jest z górnego 4 kwartyla projektów
        # (zaleta jest taka, że mamy już obliczoną jego jakość)
        elif input_pattern_generation in ['3rd_quartile', '90th_percentile', 'best_n']:
            _, consolidated_data = obtain_replay_folder_contents(settings_ai)
            
            candidate_designs = []
            for entry in consolidated_data:
                candidate_designs.append({
                    'design_diffusion':np.max(entry['episode_diffusions']),
                    'pattern':entry['episode_best_design']})
            
            candidate_designs = sorted(candidate_designs, key=lambda x:x['design_diffusion'],reverse=True)
            
            if input_pattern_generation == '3rd_quartile':
                max_idx   = len(candidate_designs)//4
            elif input_pattern_generation == 'best_n':
                max_idx   = n_best_designs
            else:
                max_idx   = len(candidate_designs)//10
            
            chosen_id = np.random.randint(0, max_idx)
            
            old_diffusion   = candidate_designs[chosen_id]['design_diffusion']
            current_pattern = candidate_designs[chosen_id]['pattern']
            parameterized_current_pattern = single_shot_schroeder_parameterize(current_pattern/settings_sim['diffuser_geometry']['num_element_height_levels'])
        else:
            raise RuntimeError('wskazano zły tryb pracy algorytmu uczącego agenta')
        
        
        # Interakcja z otoczeniem
        print('')
        # Lista nagród uzyskanych w danym epizodzie
        episode_rewards    = []
        episode_diffusions = []
        
        # Najlepsza uzyskana nagroda i najlepszy projekt
        # dyfuzora z epizodu
        best_diffusion      = old_diffusion
        episode_best_design = current_pattern
        
        # Wewnętrzna pętla interakcji ze środowiskiem
        # (uczenie ma miejsce po jej wykonaniu)
        done  = False # flaga służąca do ograniczenia liczby iteracji
        score = 0     # wartość nagrody za cały epizod
        replay_transitions = []
        for step_idx in range(MAX_ITER):
            
            # Najpierw wybieramy akcję za pomocą świeżo wczytanego modelu agenta
            print('\t krok optymalizacji nr:', step_idx+1,'\n')
            action_dict = agent.choose_action(parameterized_current_pattern)
            
            # Aplikujemy zmianę wyliczoną przez agenta do początkowego patternu dyfuzora
            # new_pattern[action['coords'][0],action['coords'][1]] += action['change']
            new_pattern  = cp.copy(current_pattern)
            new_pattern += (action_dict['actions_mtx'])
            
            # Jeżeli wykroczymy poza dozwolone wartości rozmiaru elementów dyfuzora 
            # to aplikujemy clipping
            max_val_allowed = settings_sim['diffuser_geometry']['num_element_height_levels']
            new_pattern[new_pattern>max_val_allowed] = max_val_allowed
            new_pattern[new_pattern<0] = 0
            
            # Wyliczamy jakość dyfuzora po aplikacji zmiany wyliczonej przez agenta
            # (i po przeprowadzeniu clippingu)
            new_diffusion = evaluate_design(settings_sim, new_pattern, reference_data)
            
            # Jeżeli efekt jest lepszy niż to co do tej pory mamy, to
            # uaktualniamy zmienne przechowujące dane o najlepszym 
            # rezultacie.
            if new_diffusion > best_diffusion:
                best_diffusion      = new_diffusion
                episode_best_design = new_pattern
            
            # Obliczamy nagrodę - czyli zmianę wsp. dyfuzji
            print('')
            reward = new_diffusion - old_diffusion
            
            # Wypisz efekt aplikacji akcji z obecnego epizodu
            print(f'old diffusion: {old_diffusion}, new diffusion: {new_diffusion}, reward: {reward}')
            
            # Jeżeli liczba iteracji doszła do wartości MAX_ITER, to
            # kończymy wykonanie 
            done = False
            if step_idx >= MAX_ITER-1: done = True
            
            parameterized_new_pattern = single_shot_schroeder_parameterize(new_pattern/settings_sim['diffuser_geometry']['num_element_height_levels'])
            
            # Tworzymy strukturę danych opisującą przejście stanów (na potrzeby
            # uczenia maszynowego).
            
            transition = {
                'current_pattern':current_pattern, 
                'parameterized_current_pattern':parameterized_current_pattern,
                'new_pattern':new_pattern,
                'parameterized_new_pattern':parameterized_new_pattern,
                'action_dict':action_dict,
                'reward':reward,
                'done':done}
            
            # Zapisujemy dane historyczne z epizodu
            episode_rewards.append(reward)
            episode_diffusions.append(new_diffusion)
            
            # Zapisujemy dane o przejściu stanów do pamięci agenta
            agent.store_transition(parameterized_current_pattern, action_dict['probs_matrix'], reward)
            
            replay_transitions.append(transition)
            
            # Uaktualniamy dane o nagrodach dla obeznego epizodu
            
            score += reward
            current_pattern               = new_pattern
            parameterized_current_pattern = parameterized_new_pattern
            old_diffusion                = new_diffusion
            
            print()
            print(current_pattern)
            print()
        
        output_dict={}
        output_dict.update({'agent_gpu':settings_ai['reinforcement_learning']['agent_gpu']})
        output_dict.update({'agent_id':settings_ai['reinforcement_learning']['agent_id']})
        output_dict.update({'episode_rewards':episode_rewards})
        output_dict.update({'episode_diffusions':episode_diffusions})
        output_dict.update({'best_diffusion':best_diffusion})
        output_dict.update({'episode_best_design':episode_best_design})
        output_dict.update({'timestamp':time.time()})
        output_dict.update({'replay_transitions':replay_transitions})
        output_dict.update({'input_pattern_generation':input_pattern_generation})
        np.save(os.path.join(replay_memory_dir,'%s.npy'%(create_timestamp())),[output_dict])
        
        # Po zakończeniu iteracji pętli wewnętrznej zapisujemy dane historyczne
        score_history.append(score)
        # Oraz uruchamiamy proces uczenia agenta na podstawie zebranych przez niego
        # doświadczeń
        agent.learn(10)
        
        # Następnie wypisujemy informacje o procesie treningu na ekranie
        print('epizod ',episode_number,'wynik %.8f' % score,
            'średni wynik %.8f' % np.mean(score_history[-100:]))
        
        # Na koniec zapisujemy dane treningu na dysk:
        episode_number += 1
        
        # zapis modelu
        rl_agent_wights_saving_path = os.path.join(settings_ai['reinforcement_learning']['rl_agent_weights_dir'],'rl_agent')
        agent.save_model(rl_agent_wights_saving_path)
        
        # rekonsolidacja pamięci powtórek
        replay_consolidation_frequency = settings_ai['reinforcement_learning']['replay_consolidation_frequency']
        if (episode_number%replay_consolidation_frequency == 0):
            if settings_ai['reinforcement_learning']['allow_replay_consolidation']:
                consolidate_replay_memory(settings_ai)