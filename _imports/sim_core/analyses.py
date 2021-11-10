# ---------------------------------------------------- #
# Analizy FDTD potrzebne dla symulatora dyfuzorów
# autor: Adam Kurowski
# data:   27.01.2021
# e-mail: akkurowski@gmail.com
# ---------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
from   scipy import signal
from   _imports.sim_core.utils import *
from   _imports.sim_core.Kgen import *
from   _imports.sim_core.fdtd_engine import *

def run_simulation_for_pattern(pattern, settings, mode='full_procedure'):
    
    if (pattern is None) and (mode != 'reference_only'):
        raise RuntimeError('W funkcji run_simulation_for_pattern parametr "pattern" nie może mieć wartości "None" jeżeli jest zmienna "mode" nie ma wartości "reference_only".')
    
    # https://cnx.org/contents/8FwHJuoc@7/Pr%C4%99dko%C5%9B%C4%87-d%C5%BAwi%C4%99ku
    
    fs = settings['basic']['fs']
    
    # Obliczenie parametrów ośrodka
    T_air_C        = settings['propagation_medium']['T_air_C']
    p_air_hPa      = settings['propagation_medium']['p_air_hPa']
    RH             = settings['propagation_medium']['RH']
    courant_number = settings['basic']['courant_number']
    
    c, Z_air       = get_air_properties(T_air_C, p_air_hPa, RH) # [m/s], [rayl]
    T              = 1/fs # [s]
    X              = c*T/courant_number # [m]
    
    domain_dimensions = settings['room_geometry']['domain_dimensions']
    wall_distance     = settings['room_geometry']['wall_distance']
    meas_radius       = settings['room_geometry']['meas_radius']
    scale_geometries  = settings['room_geometry']['scale_geometries']
    sim_duration      = settings['basic']['sim_duration']
    
    diffuser_pattern_size    = settings['diffuser_geometry']['diffuser_pattern_size']
    basic_element_dimensions = settings['diffuser_geometry']['basic_element_dimensions']
    diff_width  = basic_element_dimensions*diffuser_pattern_size[0]
    diff_height = basic_element_dimensions*diffuser_pattern_size[1]
    
    source_position   = [domain_dimensions[0]/2, domain_dimensions[1]-wall_distance, domain_dimensions[2]/2] # [m]
    shape_pos         = [(domain_dimensions[0]-diff_width*scale_geometries)/2,wall_distance,(domain_dimensions[0]-diff_height*scale_geometries)/2]
    
    num_element_height_levels = settings['diffuser_geometry']['num_element_height_levels']
    diffuser_depth            = settings['diffuser_geometry']['diffuser_depth']
    
    meas_radius_disc  = cont2disc(meas_radius,X)
    shape_pos_disc    = cont2disc(shape_pos,X)
    domain_dims_disc  = cont2disc(domain_dimensions,X)+2
    src_pos_disc      = cont2disc(source_position,X)
    sim_duration_disc = cont2disc(sim_duration,T)
    
    if settings['propagation_medium']['show_sim_params_desc']:
        print()
        print(f"Parametry wejściowe ośrodka:")
        print(f"\temperatura powietrza    T = {T_air_C} st. C")
        print(f"\tciśnienie atmosferyczne p = {p_air_hPa} hPa")
        print(f"\twilgotność względna    RH = {RH*100}%")
        print(f"\nWyliczone własności falowe ośrodka:")
        print(f"\tprędkość propagacji fali akustycznej   c     = {c} m/s")
        print(f"\timpedancja charakterystyczna powietrza Z_air = {Z_air} raili")
        print(f"\nWłasności symulacji:")
        print(f"\tkrok czasowy      T = {T} s")
        print(f"\tkrok przestrzenny X = {X} m")
        print()
    
    # Przykład pomiaru charakterystyk rozproszenia:
    
    imp_res_set_empty   = None
    imp_res_set_plate   = None
    imp_res_set_skyline = None
    
    if mode in ['full_procedure', 'reference_only']:
        imp_res_set_empty   = simulate_room(
            None, shape_pos_disc, meas_radius_disc, 
            courant_number, c,Z_air,T,X,domain_dims_disc,
            src_pos_disc,sim_duration_disc, settings)
            
        shape_plate   = generate_plate(cont2disc(diff_width*scale_geometries,X),cont2disc(basic_element_dimensions,X),cont2disc(diff_height*scale_geometries,X))
        imp_res_set_plate   = simulate_room(
            shape_plate, shape_pos_disc, meas_radius_disc, 
            courant_number, c,Z_air,T,X,domain_dims_disc,
            src_pos_disc,sim_duration_disc, settings)
    
    if mode in ['full_procedure', 'shape_only']:
        shape_skyline = generate_2D_Skyline_diffuser(
            pattern,
            element_seg_depth=cont2disc(diffuser_depth*scale_geometries/num_element_height_levels,X), 
            element_size=cont2disc(basic_element_dimensions*scale_geometries,X))
        imp_res_set_skyline = simulate_room(
            shape_skyline, shape_pos_disc, meas_radius_disc, 
            courant_number, c,Z_air,T,X,domain_dims_disc,
            src_pos_disc,sim_duration_disc, settings)
    
    if not (mode in ['full_procedure', 'shape_only', 'reference_only']):
        raise RuntimeError(f'Wybrano zły tryb pracy dla funkcji run_simulation_for_pattern ({mode})')
    
    return imp_res_set_empty, imp_res_set_plate, imp_res_set_skyline

def simulate_room(shape, shape_pos_disc, meas_radius_disc, courant_number,c,Z_air,T,X,domain_dims_disc,src_pos_disc,sim_duration_disc, settings):
    
    fs = settings['basic']['fs']
    max_impulse_frequency = settings['excitation_signal']['max_impulse_frequency']
    impulse_delay    = settings['excitation_signal']['impulse_delay']
    rigid_admittance = settings['room_geometry']['rigid_admittance']
    img_scale   = settings['visualization_2D']['img_scale']
    frame_skip  = settings['visualization_3D']['frame_skip']
    PML_damping = settings['room_geometry']['PML_damping']
    PML_width=settings['room_geometry']['PML_width']
    
    # Funkcja pobudzenia:
    excitation_signal = synthesize_impulse(max_impulse_frequency, fs, impulse_delay, sim_duration_disc)
    
    # Macierz warunków brzegowych
    K           = make_computational_domain(*domain_dims_disc)
    
    # Macierz admitancji powierzchni
    # beta        = make_beta(K, 1/Z_air, all_walls_init=0.8/Z_air, beta_wall_left=rigid_admittance)
    beta        = make_beta(K, 1/Z_air, all_walls_init=1/Z_air)
    
    if shape is not None:
        K, beta = embed_shape_into_domain(K, beta, shape, rigid_admittance, shape_pos_disc[0],shape_pos_disc[1],shape_pos_disc[2])
    
    # Wygenerowanie 37 punktów pomiarowych (umieszczonych na półokręgu)
    angles, measurement_points = semicircular_measurement_grid(center_point=(domain_dims_disc[0]//2,shape_pos_disc[1]),radius=meas_radius_disc)
    
    # operate_in_3D = True
    operate_in_3D = False
    
    show_preview = not settings['visualization_2D']['disable_visualization_2D']
    
    # Uruchomienie symulacji FDTD
    impres_set = None
    if not operate_in_3D:
        impres_set = run_fdtd(K, beta, src_pos_disc, excitation_signal, sim_duration_disc, courant_number, X, T, img_scale=img_scale,measurement_points_xy=measurement_points,measurement_points_yz=measurement_points,PML_damping=PML_damping,show_preview=show_preview, GPU_device_id = settings['basic']['GPU_device_id'],PML_width=PML_width)
    else:
        run_fdtd_render3D(K, beta, src_pos_disc, excitation_signal, sim_duration_disc, courant_number, X, T, frame_skip=frame_skip, ofname=None,PML_damping=PML_damping)
    
    return impres_set