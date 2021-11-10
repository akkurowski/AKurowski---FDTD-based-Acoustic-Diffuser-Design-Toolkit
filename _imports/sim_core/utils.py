# ---------------------------------------------------- #
# Pomocnicze procedury dla symulacji
# autor: Adam Kurowski
# data:   27.01.2021
# e-mail: akkurowski@gmail.com
# ---------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
from   scipy import signal
from scipy.signal import butter, lfilter

def fast_correlate(signal_A, signal_B):
    return signal.fftconvolve(np.matrix(signal_A), np.fliplr(np.matrix(signal_B)))[0]

def reflection_coefficient_to_admittance(R):
    return (1-R)/(1+R)

def cont2disc(continuous_measure, step):
    return np.round(continuous_measure/step).astype(int)

def get_air_speed(p_air_hPa, rho_air):
    # Adiabatic index or ratio of specific heats κ (kappa) = cp / cv = 1.402 for air.
    # https://cnx.org/contents/8FwHJuoc@7/Pr%C4%99dko%C5%9B%C4%87-d%C5%BAwi%C4%99ku
    # http://www.sengpielaudio.com/calculator-speedsound.htm
    # https://en.wikipedia.org/wiki/Speed_of_sound
    # cytowanie - wykłady Feynmanna, tom 1.2
    kappa    = 1.402
    p_air_Pa = p_air_hPa*100 # Pa
    c = np.sqrt(kappa*p_air_Pa/rho_air)
    return c

def get_air_density(T_air_C, p_air_hPa, RH):
    # https://pl.wikipedia.org/wiki/Stan_standardowy
    r_air        = 287.05  # J/(kg * K) powietrze
    r_wv         = 461.495 # J/(kg * K) woda
    p_wet_air_Pa = p_air_hPa*100 # Pa
    T_air_K      = T_air_C + 273.15 # deg. K
    
    # Ciśnienie nasycenia pary wodnej (przybliżone, wzór Tetena)
    p_wv_sat = 610.78*np.power(10,(7.5*T_air_C)/(T_air_C+237.3))
    
    p_wv         = RH*p_wv_sat
    p_dry_air_Pa = p_wet_air_Pa - p_wv
    
    # Obliczenie gęstości według wzoru CIPM-2007
    # A. Picard, R. S. Davis, M. Gläser, and K. Fujii, “Revised formula for the density of moist air (CIPM-2007),” Metrologia, vol. 45, no. 2, pp. 149–155, 2008, doi: 10.1088/0026-1394/45/2/004.
    
    rho_air = p_dry_air_Pa/(r_air*T_air_K) + p_wv/(r_wv*T_air_K)
    
    # Źródło:
    # https://pl.wikipedia.org/wiki/G%C4%99sto%C5%9B%C4%87_powietrza
    # rho_air = p_air_Pa/(r_air*T_air_K) # gęstość powietrza, kg/m^3
    return rho_air

# Pozyskanie szybkości propagacji dźwieku i gęstości 
# powietrza na podstawie fizycznych własności powietrza.
def get_air_properties(T_air_C=25, p_air_hPa=1013.25, RH=0.5):
    rho_air = get_air_density(T_air_C, p_air_hPa, RH)
    c       = get_air_speed(p_air_hPa, rho_air)
    Z_air   = c*rho_air # Rayls
    return c, Z_air

# Synteza pobudzenia impulsowego
def synthesize_impulse(max_freq, fs, init_delay, signal_length):
    
    # Przeliczenie parametrów wejściowych
    T           = 1/fs
    fc          = max_freq/fs
    sigma       = np.sqrt(2*np.log(2))/(2*np.pi*(fc/T))
    n           = np.arange(0,signal_length)
    
    # Synteza surowej postaci impulsu
    excitation_signal = np.exp(-np.square(T)*np.square((n-init_delay))/(2*np.square(sigma)))
    
    # Pomiar maksymalnej wartości przed filtrowaniem 
    # składowej stałej
    max_val     = np.max(np.abs(excitation_signal))
    
    # Filtracja składowej stałej
    excitation_signal       = signal.lfilter([1, -1], [1, -0.995], excitation_signal)
    
    # Normalizacja amplitudy pobudzenia
    excitation_signal = excitation_signal/np.max(np.abs(excitation_signal))*max_val
    
    return excitation_signal

# Synteza punktów pomiarowych ułożonych na półokręgu
def semicircular_measurement_grid(center_point=(0,0),radius=1,n_points=37, angles = [-np.pi/2,np.pi/2]):
    
    measurement_points = []
    angles       = np.linspace(angles[0],angles[1],n_points)
    for angle in angles:
        point = np.array([np.sin(angle),np.cos(angle)])*radius
        point += center_point
        measurement_points.append(point.tolist())
    return angles, measurement_points

def wideband_polar_response(responses_set):
    polar_resp = []
    for raw_impres in responses_set:
        polar_resp.append(np.mean(np.power(raw_impres,2)))
    polar_resp = np.array(polar_resp)
    return polar_resp

def norm_response(response):
    res_norm = response/np.max(response)
    res_db   = 20*np.log10(res_norm)
    return res_db

def draw_polar_response(polar_responses, angles=None, labels=None):
    if type(polar_responses) != list:
        polar_responses = [polar_responses]
    
    if labels is None:
        labels = [None]*len(polar_responses)
    
    if angles is None:
        angles = np.linspace(-np.pi/2,np.pi/2,len(polar_responses[0]))
    
    plt.figure()
    for res, lab in zip(polar_responses,labels):
        norm_sig = norm_response(res)
        plt.polar(angles,norm_sig, label=lab)
    plt.gca().set_theta_offset(1/2*np.pi)
    plt.gca().set_thetamin(-90)
    plt.gca().set_thetamax(90)
    plt.legend()

def generate_octave_bands(bands_f0):
    bands_boundaries = []
    for f0 in bands_f0:
        bands_boundaries.append([f0/np.sqrt(2),f0*np.sqrt(2)])
    return bands_boundaries

def subband_polar_response(b, a,imp_res_set):
    band_polar_resp    = []
    for imp_res in imp_res_set:
        imp_bnd_res    = lfilter(b, a, imp_res)
        imp_band_power = np.mean(np.power(imp_bnd_res,2))
        band_polar_resp.append(imp_band_power)
    band_polar_resp    = np.array(band_polar_resp)
    return band_polar_resp

def get_subband_polar_responses(settings, impulse_responses, filter_defs=None, order = 3):
    bands_f0        = settings['basic']['bands_f0']
    fs              = settings['basic']['fs']
    band_boundaries = generate_octave_bands(bands_f0)
    
    polar_responses = []
    for band_bounds, f0 in zip(band_boundaries, bands_f0):
        if filter_defs is None:
            b, a = butter(order, [band_bounds[0]/(fs*0.5), band_bounds[1]/(fs*0.5)], btype='band')
        else:
            b = filter_defs[f0]['b']
            a = filter_defs[f0]['a']
        band_polar_resp = subband_polar_response(b, a,impulse_responses)
        polar_responses.append(band_polar_resp)
    
    return polar_responses

def draw_subband_polar_response(settings, imp_responses, filter_defs=None):
    bands_f0        = settings['basic']['bands_f0']
    polar_responses = get_subband_polar_responses(settings, imp_responses, filter_defs=filter_defs)
    draw_polar_response(polar_responses, labels=bands_f0)

def unnorm_diffusion_coeff(polar_response):
    numerator   = np.power(np.sum(polar_response),2)
    numerator  -= np.sum(np.power(polar_response,2))
    denominator = (len(polar_response)-1)*np.sum(np.power(polar_response,2))
    diff_coeff  = numerator/denominator
    return diff_coeff

def norm_diffusion_coeff(polar_response_obj,polar_response_plt):
    diff_coeff_obj  = unnorm_diffusion_coeff(polar_response_obj)
    diff_coeff_plt  = unnorm_diffusion_coeff(polar_response_plt)
    norm_diff_coeff = (diff_coeff_obj-diff_coeff_plt)/(1-diff_coeff_plt)
    
    if norm_diff_coeff < 0:
        norm_diff_coeff = 0
    
    return norm_diff_coeff

def obtain_diffusion_coeffs(settings, obj_responses, plate_responses, filter_defs=None):
    
    bands_f0               = settings['basic']['bands_f0']
    polar_responses_obj    = get_subband_polar_responses(settings, obj_responses, filter_defs=filter_defs)
    polar_responses_plt    = get_subband_polar_responses(settings, plate_responses, filter_defs=filter_defs)
    diffusion_coefficients = []
    
    for i in range(len(bands_f0)):
        norm_diff_coeff = norm_diffusion_coeff(polar_responses_obj[i],polar_responses_plt[i])
        diffusion_coefficients.append(norm_diff_coeff)
    
    return diffusion_coefficients

def get_filter_defs(settings, order=2):
    bands_f0        = settings['basic']['bands_f0']
    fs              = settings['basic']['fs']
    band_boundaries = generate_octave_bands(bands_f0)
    filter_defs     = {}
    for band_bounds, f0 in zip(band_boundaries, bands_f0):
        filter_defs.update({f0:{}})
        b, a = butter(order, [band_bounds[0]/(fs*0.5), band_bounds[1]/(fs*0.5)], btype='band')
        filter_defs[f0]['b'] = b
        filter_defs[f0]['a'] = a
    return filter_defs

def wideband_diffusion_coefficients(reference_data, shape_data, ignore_room=False):
    
    if ignore_room:
        impres_obj_xy   = shape_data['object'][0]
        impres_plate_xy = reference_data['plate'][0]
        impres_obj_yz   = shape_data['object'][1]
        impres_plate_yz = reference_data['plate'][1]
    else:
        impres_obj_xy   = shape_data['object'][0]-reference_data['room'][0] 
        impres_plate_xy = reference_data['plate'][0]-reference_data['room'][0] 
        impres_obj_yz   = shape_data['object'][1]-reference_data['room'][1] 
        impres_plate_yz = reference_data['plate'][1]-reference_data['room'][1] 
    
    impres_obj_xy   = shape_data['object'][0]     
    impres_plate_xy = reference_data['plate'][0] 
    impres_obj_yz   = shape_data['object'][1]    
    impres_plate_yz = reference_data['plate'][1]
    
    wpr_obj_xy = wideband_polar_response(impres_obj_xy)
    wpr_plt_xy = wideband_polar_response(impres_plate_xy)
    wpr_obj_yz = wideband_polar_response(impres_obj_yz)
    wpr_plt_yz = wideband_polar_response(impres_plate_yz)
    
    wideband_diff_coeffs = {
        'plane_xy':None,
        'plane_yz':None
        }
    
    wideband_diff_coeffs['plane_xy'] = norm_diffusion_coeff(wpr_obj_xy,wpr_plt_xy)
    wideband_diff_coeffs['plane_yz'] = norm_diffusion_coeff(wpr_obj_yz,wpr_plt_yz)
    
    mean_wdbnd_diff = (wideband_diff_coeffs['plane_xy'] + wideband_diff_coeffs['plane_yz'])/2
    print(f"wyliczona dyfuzja w pł. xy: {wideband_diff_coeffs['plane_xy']}, yz: {wideband_diff_coeffs['plane_yz']}")
    return wideband_diff_coeffs, mean_wdbnd_diff

def diffusion_reward_value(settings, reference_data, shape_data, filter_defs=None, ignore_room=False):
    
    if ignore_room:
        impres_obj_xy   = shape_data['object'][0]
        impres_plate_xy = reference_data['plate'][0]
        impres_obj_yz   = shape_data['object'][1]
        impres_plate_yz = reference_data['plate'][1]
    else:
        impres_obj_xy   = shape_data['object'][0]-reference_data['room'][0] 
        impres_plate_xy = reference_data['plate'][0]-reference_data['room'][0] 
        impres_obj_yz   = shape_data['object'][1]-reference_data['room'][1] 
        impres_plate_yz = reference_data['plate'][1]-reference_data['room'][1] 
    
    diffcf_xy = obtain_diffusion_coeffs(settings, impres_obj_xy, impres_plate_xy, filter_defs=filter_defs)
    diffcf_yz = obtain_diffusion_coeffs(settings, impres_obj_yz, impres_plate_yz, filter_defs=filter_defs)
    
    wideband_diff_coeffs = {
        'plane_xy':None,
        'plane_yz':None
        }
    wpr_obj_xy = wideband_polar_response(impres_obj_xy)
    wpr_plt_xy = wideband_polar_response(impres_plate_xy)
    wpr_obj_yz = wideband_polar_response(impres_obj_yz)
    wpr_plt_yz = wideband_polar_response(impres_plate_yz)
    wideband_diff_coeffs['plane_xy'] = norm_diffusion_coeff(wpr_obj_xy,wpr_plt_xy)
    wideband_diff_coeffs['plane_yz'] = norm_diffusion_coeff(wpr_obj_yz,wpr_plt_yz)
    
    bandpass_diff_coeffs = {
        'plane_xy':{},
        'plane_yz':{}
        }
    bands_f0 = settings['basic']['bands_f0']
    for i, f0 in enumerate(bands_f0):
        bandpass_diff_coeffs['plane_xy'].update({f0:diffcf_xy[i]})
        bandpass_diff_coeffs['plane_yz'].update({f0:diffcf_yz[i]})
    
    reward_value = []
    for dc_xy, dc_yz in zip(diffcf_xy,diffcf_yz):
        reward_value += [np.min([dc_xy, dc_yz])]
    reward_value = np.mean(reward_value)
    
    return wideband_diff_coeffs, bandpass_diff_coeffs, reward_value