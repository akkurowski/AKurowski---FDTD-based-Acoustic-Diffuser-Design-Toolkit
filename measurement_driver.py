# ---------------------------------------------------- #
# Post-processing driver for data obtained from the
# anechoic-condition measurements
# autor: Adam Kurowski
# data:   17.02.2021
# e-mail: akkurowski@gmail.com
# ---------------------------------------------------- #

from _imports import *
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import scipy.io.wavfile as wave
from   scipy import signal
import plotly.express as px

# ---------------------------------------------------- #

if os.name == 'nt':
    os.system('cls')
else:
    os.system('clear')

# konfiguracja procedur symulacyjnych
CONFIG_PATH_SIM = '_settings/sim_default.ini'
settings_sim    = read_config(CONFIG_PATH_SIM)

# ---------------------------------------------------- #
# nastawy programu

measurement_data_dir          = os.path.join('..','_measurement_data')
post_processed_data_dir       = '_stored_data'
impulse_responses_output_path = os.path.join(post_processed_data_dir,'raw_impulse_responses.npy')

def ms2samples(val_ms, fs):
    return int(val_ms*0.001*fs)

# przedział próbek, do którego przycinane są odpowiedzi impulsowe po ich obliczeniu
fs                     = 48000

# sprawdzony przedział - od 1.2 ms do 10 ms
# liczyć z odrzuceniem odejmowania pomieszcenia
start_sample           = ms2samples(1.2, fs)
end_sample             = ms2samples(10, fs)

# przedział do wizualizacji
# start_sample           = ms2samples(0, fs)
# end_sample             = ms2samples(15, fs)

clipping_samples_range = [start_sample,end_sample]

# indeks próbki, która będzie zawierać maksimum sgnału
# (względem niej będzie dokonywane wyrównywanie czasowe sygnału)
desired_maximum_sample_idx = 20

# ---------------------------------------------------- #
# Narysowanie nagłówka programu

os.system('cls')
print()
print('--------------------------------------------------')
print(' Procesor danych pomiarowych (war. bezechowe)')
print(' autor:         A. Kurowski')
print(' e-mail:        akkurowski@gmail.com')
print(' wersja z dnia: 17.02.2021')
print('--------------------------------------------------')
print()

# ---------------------------------------------------- #
# Procedury pomocnicze

def get_impulse_responses_set_from_dir(scenario_dir,clipping_samples_range, desired_maximum_sample_idx):

    impulse_responses_set = []
    
    print(f"przetwarzanie: {scenario_dir}")
    for fname in os.listdir(scenario_dir):
        _, ext = os.path.splitext(fname)
        if ext !='.dat': continue
        
        path       = os.path.join(scenario_dir,fname)
        data       = np.fromfile(path, dtype=np.dtype("<f4"))
        excitation = data[0::5]
        pressure   = data[1::5]
        
        # wyodrębnienie odpowiedzi impulsowej
        impulse_response_pressure = fast_correlate(pressure, excitation)
        
        
        # wyrównanie występowania maksimów odpowiedzi
        imp_res_argmax = np.argmax(impulse_response_pressure)
        shift_value    = -(imp_res_argmax - desired_maximum_sample_idx)
        
        # Przesuwamy odpowiedź i zerujemy próbki, które "wyszły poza nawias"
        impulse_response_pressure = np.roll(impulse_response_pressure, shift_value)
        
        if (shift_value<0):
            impulse_response_pressure[-shift_value:len(impulse_response_pressure)] = 0
        
        if (shift_value>0):
            impulse_response_pressure[0:shift_value] = 0
        
        # przycięcie odpowiedzi impulsowej
        impulse_response_pressure = impulse_response_pressure[clipping_samples_range[0]:clipping_samples_range[1]]
        
        # zamieszenie odpowiedzi w wynikowej liście
        impulse_responses_set.append(impulse_response_pressure)
        
    
    return impulse_responses_set

def extract_impulse_responses(input_dir, output_dir, clipping_samples_range, desired_maximum_sample_idx):
    
    print()
    set_output_dict = {}
    set_output_dict.update({'room':       get_impulse_responses_set_from_dir(os.path.join(input_dir,'0_pomieszczenie'), clipping_samples_range, desired_maximum_sample_idx)})
    set_output_dict.update({'plate':      get_impulse_responses_set_from_dir(os.path.join(input_dir,'1_plyta'),         clipping_samples_range, desired_maximum_sample_idx)})
    set_output_dict.update({'genetic_xy': get_impulse_responses_set_from_dir(os.path.join(input_dir,'2_genetyczne_xy'), clipping_samples_range, desired_maximum_sample_idx)})
    set_output_dict.update({'genetic_yz': get_impulse_responses_set_from_dir(os.path.join(input_dir,'3_genetyczne_yz'), clipping_samples_range, desired_maximum_sample_idx)})
    set_output_dict.update({'dpg_xy':     get_impulse_responses_set_from_dir(os.path.join(input_dir,'4_dpg_xy'),        clipping_samples_range, desired_maximum_sample_idx)})
    set_output_dict.update({'dpg_yz':     get_impulse_responses_set_from_dir(os.path.join(input_dir,'5_dpg_yz'),        clipping_samples_range, desired_maximum_sample_idx)})
    set_output_dict.update({'random_xy':  get_impulse_responses_set_from_dir(os.path.join(input_dir,'6_random_xy'),     clipping_samples_range, desired_maximum_sample_idx)})
    set_output_dict.update({'random_yz':  get_impulse_responses_set_from_dir(os.path.join(input_dir,'7_random_yz'),     clipping_samples_range, desired_maximum_sample_idx)})
    print()
    np.save(output_dir,set_output_dict)

def filter_scenario_impulse_responses(object_structure, f_lo=50, f_hi=4200):
    
    order = 5
    b, a  = butter(order, [f_lo/(fs*0.5), f_hi/(fs*0.5)], btype='band')
    
    def filter_imp_res_set(b, a, imp_res_set):
        filtered_imp_res_set = []
        for imp_res in imp_res_set:
            filtered_imp_res_set.append(signal.lfilter(b,a,imp_res))
        filtered_imp_res_set = np.array(filtered_imp_res_set)
        return filtered_imp_res_set
    
    for plane_idx in [0,1]:
        if 'plate' in list(object_structure.keys()):
            imp_res_set = object_structure['plate'][plane_idx]
            object_structure['plate'][plane_idx]  = filter_imp_res_set(b, a, object_structure['plate'][plane_idx])
        if 'room' in list(object_structure.keys()):
            object_structure['room'][plane_idx]   = filter_imp_res_set(b, a, object_structure['room'][plane_idx])
        if 'object' in list(object_structure.keys()):
            object_structure['object'][plane_idx] = filter_imp_res_set(b, a, object_structure['object'][plane_idx])
    
    return object_structure

def obtain_impulse_responses(impulse_responses_dir):
    
        impulse_responses_dict = np.load(impulse_responses_dir, allow_pickle=True).item()
    
        reference_data = {}
        reference_data.update({'plate':[
            np.array(impulse_responses_dict['plate']),
            np.array(impulse_responses_dict['plate'])]})
        
        reference_data.update({'room':[
            np.array(impulse_responses_dict['room']),
            np.array(impulse_responses_dict['room'])]})
        
        diffuser_genetic = {}
        diffuser_genetic.update({'object':[
            np.array(impulse_responses_dict['genetic_xy']),
            np.array(impulse_responses_dict['genetic_yz'])]})
        
        diffuser_dpg = {}
        diffuser_dpg.update({'object':[
            np.array(impulse_responses_dict['dpg_xy']),
            np.array(impulse_responses_dict['dpg_yz'])]})
        
        diffuser_random = {}
        diffuser_random.update({'object':[
            np.array(impulse_responses_dict['random_xy']),
            np.array(impulse_responses_dict['random_yz'])]})
        
        f_lo = 250/np.sqrt(2)
        f_hi = 4000*np.sqrt(2)
        if ask_for_user_preference(f'Czy przefiltrować odpowiedzi do pasma symulacji ({"%.2f"%f_lo} Hz - {"%.2f"%f_hi} Hz)?'):
            reference_data   = filter_scenario_impulse_responses(reference_data,   f_lo, f_hi)
            diffuser_genetic = filter_scenario_impulse_responses(diffuser_genetic, f_lo, f_hi)
            diffuser_dpg     = filter_scenario_impulse_responses(diffuser_dpg,     f_lo, f_hi)
            diffuser_random  = filter_scenario_impulse_responses(diffuser_random,  f_lo, f_hi)
        
        return reference_data, diffuser_genetic, diffuser_dpg, diffuser_random


# ---------------------------------------------------- #
# Uruchomienie głównej pętli interakcji
while True:
    main_menu_items = []
    main_menu_items.append('wyodrębnij odpowiedzi impulsowe')
    main_menu_items.append('zwizualizuj odpowiedzi impulsowe')
    main_menu_items.append('zwizualizuj odpowiedzi impulsowe po odjęciu wpływu pomieszczenia')
    main_menu_items.append('rysuj szerokopasmowe charakterystyki dyfuzorów')
    main_menu_items.append('oblicz szerokopasmowe parametry dyfuzorów')
    main_menu_items.append('rysuj wąskopasmowe charakterystyki dyfuzorów')
    main_menu_items.append('rysuj wąskopasmowe wsp. dyfuzji dyfuzorów')
    main_menu_items.append('wyjście')
    
    mm_ans = ask_user_for_an_option_choice('\nWybierz nr akcji do wykonania', 'Nr akcji:', main_menu_items)
    
    if mm_ans == 'wyodrębnij odpowiedzi impulsowe':
        extract_impulse_responses(measurement_data_dir, impulse_responses_output_path, clipping_samples_range, desired_maximum_sample_idx)
        print('ekstrakcja odpowiedzi została zakończona')
    
    elif mm_ans == 'zwizualizuj odpowiedzi impulsowe':
        print()
        if not os.path.isfile(impulse_responses_output_path):
            print('nie znaleziono pliku z odpowiedziami impulsowymi, wygeneruj je odpowiednim poleceniem skryptu')
            continue
        
        impulse_responses_dict = np.load(impulse_responses_output_path, allow_pickle=True).item()
        
        available_scenarios = list(impulse_responses_dict.keys())
        
        visualized_scenario = ask_user_for_an_option_choice('\nWybierz nr scenariusza do zwizualizowania:', 'Nr scenariusza:', available_scenarios)
        
        imp_resp_set = impulse_responses_dict[visualized_scenario]
        plt.figure()
        for i, imp_res in enumerate(imp_resp_set):
            angle = i*5 # stopni (od prawej strony dyfuzora widzianego od frontu), dyfuzor plecami do wejścia do komory
            t_vec = np.arange(len(imp_res))/fs
            plt.plot(t_vec, imp_res, label=angle)
        
        plt.grid()
        plt.xlabel('czas [ms]')
        plt.ylabel('wart. chwilowa')
        plt.show()
            
        print()
    
    elif mm_ans == 'zwizualizuj odpowiedzi impulsowe po odjęciu wpływu pomieszczenia':
        print()
        if not os.path.isfile(impulse_responses_output_path):
            print('nie znaleziono pliku z odpowiedziami impulsowymi, wygeneruj je odpowiednim poleceniem skryptu')
            continue
        
        impulse_responses_dict = np.load(impulse_responses_output_path, allow_pickle=True).item()
        
        available_scenarios = list(impulse_responses_dict.keys())
        
        visualized_scenario = ask_user_for_an_option_choice('\nWybierz nr scenariusza do zwizualizowania:', 'Nr scenariusza:', available_scenarios)
        
        selected_imp_resps     = np.array(impulse_responses_dict[visualized_scenario])
        room_impulse_responses = np.array(impulse_responses_dict['room'])
        
        imp_resp_set = selected_imp_resps - room_impulse_responses
        plt.figure()
        for i, imp_res in enumerate(imp_resp_set):
            angle = i*5 # stopni (od prawej strony dyfuzora widzianego od frontu), dyfuzor plecami do wejścia do komory
            t_vec = np.arange(len(imp_res))/fs
            plt.plot(t_vec, imp_res, label=angle, c='black')
        
        plt.grid()
        plt.xlabel('time [ms]')
        plt.ylabel('instantaneous value [-]')
        plt.show()
            
        print()
    
    
    elif mm_ans == 'rysuj wąskopasmowe charakterystyki dyfuzorów':
        print()
        if not os.path.isfile(impulse_responses_output_path):
            print('nie znaleziono pliku z odpowiedziami impulsowymi, wygeneruj je odpowiednim poleceniem skryptu')
            continue
        
        reference_data, diffuser_genetic, diffuser_dpg, diffuser_random = obtain_impulse_responses(impulse_responses_output_path)
        
        ylim_spec = [-25,0]
        draw_subband_polar_response(settings_sim, diffuser_random['object'][0])
        plt.title('random xy')
        plt.ylim(ylim_spec)
        draw_subband_polar_response(settings_sim, diffuser_random['object'][1])
        plt.title('random yz')
        plt.ylim(ylim_spec)
        
        draw_subband_polar_response(settings_sim, diffuser_genetic['object'][0])
        plt.title('genetic xy')
        plt.ylim(ylim_spec)
        draw_subband_polar_response(settings_sim, diffuser_genetic['object'][1])
        plt.title('genetic yz')
        plt.ylim(ylim_spec)
        
        draw_subband_polar_response(settings_sim, diffuser_dpg['object'][0])
        plt.title('dpg xy')
        plt.ylim(ylim_spec)
        draw_subband_polar_response(settings_sim, diffuser_dpg['object'][1])
        plt.title('dpg yz')
        plt.ylim(ylim_spec)
        
        plt.show()
    
    elif mm_ans == 'rysuj szerokopasmowe charakterystyki dyfuzorów':
        print()
        if not os.path.isfile(impulse_responses_output_path):
            print('nie znaleziono pliku z odpowiedziami impulsowymi, wygeneruj je odpowiednim poleceniem skryptu')
            continue
        
        reference_data, diffuser_genetic, diffuser_dpg, diffuser_random = obtain_impulse_responses(impulse_responses_output_path)
        
        ylim_spec = [-25,0]
        
        draw_polar_response([
            wideband_polar_response(diffuser_genetic['object'][0]),
            wideband_polar_response(diffuser_genetic['object'][1])],
            labels = ['xy','yz'])
        plt.title('genetic')
        plt.ylim(ylim_spec)
        plt.legend()
        
        draw_polar_response([
            wideband_polar_response(diffuser_dpg['object'][0]),
            wideband_polar_response(diffuser_dpg['object'][1])],
            labels = ['xy','yz'])
        plt.title('dpg')
        plt.ylim(ylim_spec)
        plt.legend()
        
        draw_polar_response([
            wideband_polar_response(diffuser_random['object'][0]),
            wideband_polar_response(diffuser_random['object'][1])],
            labels = ['xy','yz'])
        plt.title('random')
        plt.ylim(ylim_spec)
        plt.legend()
        
        plt.show()
    
    elif mm_ans == 'rysuj wąskopasmowe wsp. dyfuzji dyfuzorów':
        print()
        if not os.path.isfile(impulse_responses_output_path):
            print('nie znaleziono pliku z odpowiedziami impulsowymi, wygeneruj je odpowiednim poleceniem skryptu')
            continue
        
        reference_data, diffuser_genetic, diffuser_dpg, diffuser_random = obtain_impulse_responses(impulse_responses_output_path)
        
        ignore_room = ask_for_user_preference('Czy zignorować pomiar pomieszczenia? (może on zawierać zakłócające pomiar odbicia)')
        
        _, bandpass_diff_coeffs_genetic, _ = diffusion_reward_value(settings_sim, reference_data, diffuser_genetic, ignore_room=ignore_room)
        _, bandpass_diff_coeffs_dpg, _     = diffusion_reward_value(settings_sim, reference_data, diffuser_dpg, ignore_room=ignore_room)
        _, bandpass_diff_coeffs_random, _  = diffusion_reward_value(settings_sim, reference_data, diffuser_random, ignore_room=ignore_room)
        
        freqs = list(bandpass_diff_coeffs_genetic['plane_xy'].keys())
        
        def draw_scenario(freqs, bandpass_diff_coeffs, scenario_name, color):
            plt.plot(freqs, bandpass_diff_coeffs['plane_xy'].values(), '-', c=color, label=f'{scenario_name} xy')
            plt.scatter(freqs, bandpass_diff_coeffs['plane_xy'].values(), c=color, label=None)
            plt.plot(freqs, bandpass_diff_coeffs['plane_yz'].values(), '--', c=color, label=f'{scenario_name} yz')
            plt.scatter(freqs, bandpass_diff_coeffs['plane_yz'].values(), c=color, label=None)
        
        plt.figure()
        draw_scenario(freqs, bandpass_diff_coeffs_genetic, 'genetic', 'darkblue')
        draw_scenario(freqs, bandpass_diff_coeffs_dpg, 'dpg', 'orange')
        draw_scenario(freqs, bandpass_diff_coeffs_random, 'random', 'darkred')
        
        plt.legend()
        plt.grid()
        plt.xlabel('frequency [Hz]')
        plt.ylabel('diffusion coefficient [-]')
        plt.gca().set_xscale('log')
        plt.xlim([125,10000])
        plt.ylim([0,1])
        plt.show()
        
    
    elif mm_ans == 'oblicz szerokopasmowe parametry dyfuzorów':
        print()
        if not os.path.isfile(impulse_responses_output_path):
            print('nie znaleziono pliku z odpowiedziami impulsowymi, wygeneruj je odpowiednim poleceniem skryptu')
            continue
        
        reference_data, diffuser_genetic, diffuser_dpg, diffuser_random = obtain_impulse_responses(impulse_responses_output_path)
        
        ignore_room = ask_for_user_preference('Czy zignorować pomiar pomieszczenia? (może on zawierać zakłócające pomiar odbicia)')
        
        print()
        print('genetic pattern')
        _, mean_diff = wideband_diffusion_coefficients(reference_data, diffuser_genetic, ignore_room=ignore_room)
        print(f"mean diffusion: {mean_diff}")
        
        print()
        print('dpg pattern')
        _, mean_diff = wideband_diffusion_coefficients(reference_data, diffuser_dpg, ignore_room=ignore_room)
        print(f"mean diffusion: {mean_diff}")
        
        print()
        print('random pattern')
        _, mean_diff = wideband_diffusion_coefficients(reference_data, diffuser_random, ignore_room=ignore_room)
        print(f"mean diffusion: {mean_diff}")
        
        print()
        
    
    elif mm_ans == 'wyjście':
        break
    
    else:
        raise RuntimeError('Wybrano zły numer akcji w menu głównym.')

