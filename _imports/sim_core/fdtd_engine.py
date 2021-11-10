# ------------------------------------------------------
# Silnik obliczeniowy dla metody FDTD z akceleracją GPU
# autor:  Adam Kurowski
# data:   25.01.2021
# e-mail: akkurowski@gmail.com
# ------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from   numba import prange, cuda, float32
from   scipy.interpolate import RectBivariateSpline
from   mayavi import mlab
import copy as cp
import os
import cv2
from numba import cuda
from   _imports.sim_core.Kgen import *

# ------------------------------------------------------
# Funkcje pomocnicze

# Funkcja sprawdzająca aktualny indeks dla którego wykonywane są 
# obliczenia - zapobiega wykonywaniu obliczeń, jeżeli indeks jest 
# poza domeną obliczeń.
@cuda.jit('boolean(int32,int32,int32,int32,int32,int32)', device=True)
def pressure_outside_domain(ll,mm,ii,sh_ll,sh_mm,sh_ii):
    if ll < 1 or ll >= sh_ll-1:
        return True
    if mm < 1 or mm >= sh_mm-1:
        return True
    if ii < 1 or ii >= sh_ii-1: 
        return True
    return False

# Krok obliczeń CUDA
@cuda.jit
def FDTD_step_CUDA(K,courant_number,X,T,sigma_arr,c,rho,p_all_steps,BK, p_current):

    cuda.syncthreads()
    ll,mm,ii = cuda.grid(3)
    if pressure_outside_domain(ll,mm,ii,*(p_current.shape)): return
    cuda.syncthreads()
    
    damping_A = rho[0]*c[0]*c[0]*sigma_arr[ll, mm, ii]*T[0] + sigma_arr[ll, mm, ii]*T[0]/rho[0]
    damping_B = c[0]*c[0]*sigma_arr[ll, mm, ii]*sigma_arr[ll, mm, ii]*T[0]*T[0]
    
    # Aplikacja równania falowego z uwzględnieniem prostopadłych warunków
    # brzegowych i współczynnika tłumienia ośrodka stratnego (dla PML).
    p_current[ll,mm,ii] = (
    (2-K[ll, mm, ii]*courant_number[0]*courant_number[0])*p_all_steps[ll,mm,ii,1] 
    + (BK[ll, mm, ii] + damping_A/2 -1)*p_all_steps[ll,mm,ii,2] 
    + courant_number[0]*courant_number[0]*(p_all_steps[ll+1,mm,ii,1] + p_all_steps[ll-1,mm,ii,1] + p_all_steps[ll,mm+1,ii,1] + p_all_steps[ll,mm-1,ii,1] + p_all_steps[ll,mm,ii+1,1] + p_all_steps[ll,mm,ii-1,1])
    ) / (1+BK[ll, mm, ii] + damping_A/2 + damping_B)

# Uaktualnienie macierzy zawierających dane rozkładów ciśnień z kolejnych kroków obliczeń
@cuda.jit
def shift_pressure_matrices_CUDA(p_all_steps):
    ll,mm,ii = cuda.grid(3)
    if pressure_outside_domain(ll,mm,ii,*(p_all_steps.shape[0:3])): return
    p_all_steps[ll,mm,ii,2] = p_all_steps[ll,mm,ii,1]
    p_all_steps[ll,mm,ii,1] = p_all_steps[ll,mm,ii,0]

# Pobranie informacji o aktualnym rozkładzie ciśnienia w symulacji
@cuda.jit
def update_pressure_CUDA(p_all_steps,p_current):
    ll,mm,ii = cuda.grid(3)
    if pressure_outside_domain(ll,mm,ii,*(p_current.shape)): return
    p_all_steps[ll,mm,ii,0] = p_current[ll,mm,ii]

# Wstrzyknięcie następnej wartości sygnału pobudzenia (soft source)
@cuda.jit
def inject_source_CUDA(p_all_steps,source_pos_disc,excitation_sample):
    ll,mm,ii = cuda.grid(3)
    if pressure_outside_domain(ll,mm,ii,*(p_all_steps.shape[0:3])): return
    p_all_steps[source_pos_disc[0],source_pos_disc[1],source_pos_disc[2],1] = p_all_steps[source_pos_disc[0],source_pos_disc[1],source_pos_disc[2],1] + excitation_sample[0];

# Wymuszenie braku ciśnienia w przestrzeniach oznaczonych w macierzy K_glMem
# wartością 0 (wnętrze obiektów).
@cuda.jit
def dead_spaces_CUDA(p_all_steps,zero_K):
    i = cuda.grid(1)
    
    # Wyłączenie funkcji, gdy indeks jest poza domeną obliczeniową.
    if i < 0 or i >= zero_K.shape[1]:
        return
    
    # Wymuszenie zerowego ciśnienia wewnątrz obiektów.
    p_all_steps[zero_K[0,i],zero_K[1,i],zero_K[2,i],0] = 0
    p_all_steps[zero_K[0,i],zero_K[1,i],zero_K[2,i],1] = 0
    p_all_steps[zero_K[0,i],zero_K[1,i],zero_K[2,i],2] = 0

@cuda.jit
def read_result_CUDA(p_all_steps,p_plane_xy,p_plane_yz,source_pos_disc):
    ll,mm,ii = cuda.grid(3)
    if pressure_outside_domain(ll,mm,ii,*(p_all_steps.shape[0:3])): return
    p_plane_xy[ll,mm] = p_all_steps[ll,mm,source_pos_disc[2],0]
    p_plane_yz[mm,ii] = p_all_steps[source_pos_disc[0],mm,ii,0]

# ------------------------------------------------------
# Solver FDTD uruchamiany na GPU, 
# udostępniony za pomocą interfejsu obiektowego
class FDTDSimulation:
    def __init__(self, K, beta, source_pos_disc, excitation_signal, courant_number, X, T, c=340, rho = 1.225, PML_damping = 0.15, 
    PML_width = 50, TPB=8):
        
        # Aktualizacja pól obiektu:
        #   geometria domeny obliczeniowej:
        self.K                 = K
        self.c                 = c
        self.rho               = rho
        self.BK                = make_BK(K,courant_number,beta)
        self.zero_K            = np.array(np.where(K==0))
        #   pobudzenie:
        self.source_pos_disc   = source_pos_disc
        self.excitation_signal = excitation_signal
        #   rozdzielczość:
        self.courant_number    = courant_number
        self.X                 = X
        self.T                 = T
        #   warstwy PML:
        self.PML_damping       = PML_damping
        self.PML_width         = PML_width
        #   dystrybucja zadań na GPU:
        self.TPB               = TPB
        #   numer klatki symulacji
        self.sim_frame_number  = 0
        
        # Alokacja macierzy ciśnień
        #   Główna zmienna zawierająca 3 kolejne rozkłady ciśnienia w domenie K
        self.p_all_steps  = np.zeros(list(self.K.shape)+[3])
        #   Rozkład 3D ciśnienia (1 krok)
        self.p_current = np.zeros(self.K.shape)
        #   Rozkład 2D ciśnienia
        self.p_plane_xy = np.zeros((self.K.shape[0],self.K.shape[1]))
        self.p_plane_yz = np.zeros((self.K.shape[1],self.K.shape[2]))
        
        # Zmienna przekazująca aktualną wartość sygnału pobudzenia
        self.excitation_sample = self.excitation_signal[0]
        
        # Alokacja przypisania wątków obliczających FDTD na GPU
        self.num_threads = [self.TPB,self.TPB,self.TPB]
        self.num_blocks  = []
        for i,n_thr in enumerate(self.num_threads):
            self.num_blocks.append(int(np.ceil(self.p_current.shape[i]/n_thr)))
        
        # Obsługa warstw PML
        self.PML_sigma       = np.zeros_like(K)
        # Profil tłumienia (tym większe tłumienie im punkt znajduje się
        # głębiej w warstwie PML).
        self.damping_profile = np.power(np.linspace(1,0,self.PML_width),1.4)*self.PML_damping
        # Aplikacja profilu PML do wszystkich granic domeny obliczeniowej.
        for i in range(self.PML_width):
            local_damping_fctr = self.damping_profile[i]
            self.PML_sigma[i,:,:]                           = local_damping_fctr
            self.PML_sigma[self.PML_sigma.shape[0]-i-1,:,:] = local_damping_fctr
            self.PML_sigma[:,i,:]                           = local_damping_fctr
            self.PML_sigma[:,self.PML_sigma.shape[1]-i-1,:] = local_damping_fctr
            self.PML_sigma[:,:,i]                           = local_damping_fctr
            self.PML_sigma[:,:,self.PML_sigma.shape[2]-i-1] = local_damping_fctr
        
        # Zmienne w pamięci GPU:
        self.K_glMem                 = cuda.to_device(self.K)
        self.c_glMem                 = cuda.to_device(np.array([self.c]))
        self.rho_glMem               = cuda.to_device(np.array([self.rho]))
        self.BK_glMem                = cuda.to_device(self.BK)
        self.zero_K_glMem            = cuda.to_device(self.zero_K)
        self.source_pos_disc_glMem   = cuda.to_device(self.source_pos_disc)
        self.excitation_sample_glMem = cuda.to_device(np.array([self.excitation_sample]))
        self.courant_number_glMem               = cuda.to_device(np.array([self.courant_number]))
        self.X_glMem                 = cuda.to_device(np.array([self.X]))
        self.T_glMem                 = cuda.to_device(np.array([self.T]))
        self.sigma_glMem             = cuda.to_device(np.array(self.PML_sigma))
        self.p_all_steps_glMem       = cuda.to_device(self.p_all_steps)
        self.p_current_glMem         = cuda.to_device(self.p_current)
        self.p_plane_xy_glMem        = cuda.to_device(self.p_plane_xy)
        self.p_plane_yz_glMem        = cuda.to_device(self.p_plane_yz)
        
    def step(self):
        # Wykonanie kroku symulacji FDTD
        FDTD_step_CUDA[self.num_blocks,self.num_threads](self.K_glMem,self.courant_number_glMem,self.X_glMem,self.T_glMem,self.sigma_glMem,self.c_glMem,self.rho_glMem,self.p_all_steps_glMem,self.BK_glMem, self.p_current_glMem)
        
        # Przypisanie wyniku kroku symulacji do macierzy ciśnienia "stanu teraźniejszego".
        update_pressure_CUDA[self.num_blocks,self.num_threads](self.p_all_steps_glMem,self.p_current_glMem)
        
        # Wymuszenie zerowego ciśnienia wewnątrz obiektów
        # Konieczne jest zdefiniowanie układu wątków dla 
        # siatki 1D charakterystycznej dla tej operacji.
        enforcer_num_threads = 32
        enforcer_num_blocks  = int(np.ceil(self.zero_K.shape[1]/32))
        dead_spaces_CUDA[enforcer_num_blocks,enforcer_num_threads](self.p_all_steps_glMem,self.zero_K_glMem)
        
        # Odczytanie rozkładu 2D ciśnienia (do późniejszego odczytu ze zmiennej p_plane).
        read_result_CUDA[self.num_blocks,self.num_threads](self.p_all_steps_glMem,self.p_plane_xy_glMem,self.p_plane_yz_glMem,self.source_pos_disc_glMem)
        
        # Wstrzyknięcie kolejnej wartości pobudzenia:
        #   wysłanie wartości pobudzenia
        self.excitation_sample_glMem.copy_to_device(np.array([self.excitation_signal[self.sim_frame_number]])) 
        self.sim_frame_number += 1
        #   uaktualnienie źródła
        inject_source_CUDA[self.num_blocks,self.num_threads](self.p_all_steps_glMem,self.source_pos_disc_glMem,self.excitation_sample_glMem) 
        
        # Przesunięcie kroków czasowych symulacji
        shift_pressure_matrices_CUDA[self.num_blocks,self.num_threads](self.p_all_steps_glMem)
    
    def get_pressure_plane(self, plane_name = 'xy'):
        # Dane o ciśnieniu z wybranej płaszczyzny 2D:
        # (wysokość z jest taka sama jak wysokość źródła)
        if plane_name == 'xy':
            self.p_plane_xy_glMem.copy_to_host(self.p_plane_xy)
            return self.p_plane_xy
        elif plane_name == 'xz':
            self.p_plane_yz_glMem.copy_to_host(self.p_plane_yz)
            return self.p_plane_yz
        elif plane_name == 'both':
            self.p_plane_xy_glMem.copy_to_host(self.p_plane_xy)
            self.p_plane_yz_glMem.copy_to_host(self.p_plane_yz)
            return self.p_plane_xy, p_plane_yz
        else:
            raise RuntimeError(f'Wybrano złą nazwę płaszczyzny do odczytania przez symulator ({plane_name})')
    
    def get_pressure_3D(self):
        # Dane o ciśnieniu z wybranej płaszczyzny 2D:
        self.p_current_glMem.copy_to_host(self.p_current)
        return self.p_current

def draw_pressure_2D_cv(K, p_plane, objects_mask, source_pos_disc, observerPosD, measurement_points, img_scale=None, axe_names=['A','B'], transpose=False):
    
    # Automatyczna regulacja wzmocnienia - aby widoczne były
    # szczegóły gdy pobudzenie ma wysokie wartości.
    ref = np.max(np.abs(p_plane))
    if ref < 1: ref = 1
    
    # Obliczenie czarno-białej wizualizacji wybranej płaszczyzny symulacji.
    bw_image = (127+(p_plane/ref)*127).astype(np.uint8)
    bw_image[objects_mask] = 255
    
    # Oznaczenie pozycji źródła (biała kropka)
    bw_image[source_pos_disc[0]-1:source_pos_disc[0]+1,source_pos_disc[1]-1:source_pos_disc[1]+1] = 255
    
    # Oznaczenie miejsc pomiaru odpowiedzi impulsowej (czarne kropki)
    for coords in measurement_points:
        
        # Jako, że koordynaty są zmiennoprzecinkowe, konieczne
        # przy wyświetlaniu będzie wprowadzenie antyaliasingu.
        cv2.circle(bw_image, (np.float32(coords[1]), np.float32(coords[0])),1,0,-1)
    
    # Upscaling wynikowego obrazka, by lepiej się go oglądało:
    if img_scale is not None:
        height, width   = bw_image.shape
        newX,newY       = bw_image.shape[1]*img_scale, bw_image.shape[0]*img_scale
        bw_image        = cv2.resize(bw_image,(int(newX),int(newY)))
    
    if transpose:
        bw_image = cv2.transpose(bw_image)
    
    lower_left_corner   = np.array([0, bw_image.shape[0]])
    first_label_coords  = tuple(lower_left_corner+np.array([5,-50]))
    second_label_coords = tuple(lower_left_corner+np.array([50,-10]))
    
    cv2.putText(bw_image, axe_names[0], first_label_coords, cv2.FONT_HERSHEY_COMPLEX_SMALL ,1,0,2)
    cv2.putText(bw_image, axe_names[1], second_label_coords, cv2.FONT_HERSHEY_COMPLEX_SMALL ,1,0,2)
    
    return bw_image

def display_pressure_2D_cv(images_list, time_str, margin=5, wnd_title="Podglad symulacji FDTD"):
    if type(images_list) != list:
        images_list = [images_list]
    
    x_sizes = []
    y_sizes = []
    for image_arr in images_list:
        x_sizes.append(image_arr.shape[0])
        y_sizes.append(image_arr.shape[1])
        
    max_x = np.max(x_sizes)
    max_y = np.max(y_sizes)
    
    for i, image_arr in enumerate(images_list):
        eff_margin = margin
        if i == len(images_list)-1: margin = 0
        new_img = np.zeros((max_x,images_list[i].shape[1]+margin)).astype(np.uint8)
        new_img[0:images_list[i].shape[0],0:images_list[i].shape[1]] = images_list[i]
        images_list[i] = new_img
    
    compound_image = np.concatenate(images_list,axis=1)
    
    cv2.putText(compound_image, f'czas: {time_str}', (20,30), cv2.FONT_HERSHEY_COMPLEX_SMALL   ,1,0,2)
    
    cv2.imshow(wnd_title,compound_image)

def sample_with_interpolation(measurement_points, p_plane):
    pressure_interpolator = RectBivariateSpline(np.arange(p_plane.shape[0]),np.arange(p_plane.shape[1]),p_plane,kx=5,ky=5)
    set_of_impres = []
    for coords in measurement_points:
        p_interpolated = pressure_interpolator(coords[0],coords[1])[0,0]
        set_of_impres.append(p_interpolated)
    return set_of_impres

# ------------------------------------------------------
# Uproszczony interfejs symulacji z podglądem 2D
def run_fdtd(K, beta, source_pos_disc, excitation_signal, sim_length, courant_number, X, T, measurement_points_xy=[], measurement_points_yz=[], show_preview = True, write_video = False, img_scale=4, ofname='last_simulation.mp4', PML_damping=0.15, PML_width=50, GPU_device_id=0):
    
    # Dla systemów z wieloma GPU - wybieramy odpowiednie
    # urządzenie do wykonania obliczeń
    # if GPU_device_id
    num_gpus = len(nb.cuda.gpus)
    
    if (GPU_device_id >= num_gpus) or (GPU_device_id < 0):
        raise RuntimeError(f'Wybrano niepoprawne id karty GPU ({GPU_device_id}, a dostępnych kart jest: {num_gpus})')
    cuda.select_device(GPU_device_id)
    
    # Zamiana osi na płaszczyźnie xz
    measurement_points_yz_tmp = []
    for coord in measurement_points_yz:
        measurement_points_yz_tmp.append([coord[1],coord[0]])
    measurement_points_yz = measurement_points_yz_tmp
    
    # Utworzenie obiektu symulacji
    sim_obj = FDTDSimulation(K, beta, source_pos_disc, excitation_signal, courant_number, X, T, PML_damping=PML_damping, PML_width=PML_width)
    
    # Obsługa zapisu wizualizacji na dysk
    if show_preview and write_video:
        video_frames   = []
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(ofname,fourcc, 30.0, (K.shape[0]*img_scale,K.shape[1]*img_scale),isColor=False)
    
    # Zmienna do której zapisywana będzie wartość
    # mierzonej odpowiedzi impulsowej
    imp_res_set_xy = []
    imp_res_set_yz = []
    
    # Główna pętla obliczająca symulację FDTD
    for nn in range(1,sim_length):
        print('%3i/%3i\r'%(nn+1,sim_length), end='')
        
        # Obliczenie kroku symulacji i pobranie danych z GPU
        sim_obj.step()
        p_plane_xy = sim_obj.get_pressure_plane('xy')
        p_plane_yz = sim_obj.get_pressure_plane('xz')
        
        # Pomiar i zapis wartości odpowiedzi impulsowych
        
        imp_res_set_xy.append(sample_with_interpolation(measurement_points_xy, p_plane_xy))
        imp_res_set_yz.append(sample_with_interpolation(measurement_points_yz, p_plane_yz))
        
        if show_preview:
            bw_images = []
            bw_images.append(draw_pressure_2D_cv(K, p_plane_xy, K[:,:,source_pos_disc[2]]==0, source_pos_disc[[0,1]], source_pos_disc[[0,1]], measurement_points_xy, img_scale=img_scale, axe_names=['x','y'], transpose=False))
            
            bw_images.append(draw_pressure_2D_cv(K, p_plane_yz, K[source_pos_disc[0],:,:]==0, source_pos_disc[[1,2]], source_pos_disc[[0,2]], measurement_points_yz, img_scale=img_scale, axe_names=['z','y'], transpose=True))
            display_pressure_2D_cv(bw_images, '%2.2f ms'%((nn*1000)*T))
            
            # Zapis wideo i wyświetlenie obrazka
            if write_video:
                out.write(bw_image)
        
            # Przy włączonym podglądzie można przerwać symulację
            # klawiszem escape.
            key = cv2.waitKey(1)
            if key == 27:
                break
    
    # Zamknięcie pliku z zapisem symulacji.
    if show_preview:        
        # Wyłączenie okna podglądu symulacji
        cv2.destroyAllWindows()
        if write_video:
            out.release()
    
    imp_res_set_xy = np.array(imp_res_set_xy).T
    imp_res_set_yz = np.array(imp_res_set_yz).T
    
    return [imp_res_set_xy,imp_res_set_yz]

# ------------------------------------------------------
# Uproszczony interfejs symulacji z renderingiem 3D
def run_fdtd_render3D(K, beta, source_pos_disc, excitation_signal, sim_length, courant_number, X, T, vis_halfspread=20, display_time = True, frame_skip=None, ofname='render_3d.avi', encode_video=False,PML_damping=25000):
    
    if ofname is not None:
        # Folder z danymi wyjściowymi
        output_folder = '_output_renders'
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
    
    # Utworzenie przetworzonej macierzy K, która
    # pozwoli na wyświetlenie obiektów rozpraszających.
    new_K = np.ones_like(K)
    new_K[2:-2,2:-2,2:-2] = K[2:-2,2:-2,2:-2]
    new_K[new_K==0] = 8
    new_K[new_K==6] = 0
    new_K[new_K==5] = 0
    
    # Utworzenie obiektu symulacji
    sim_obj = FDTDSimulation(K, beta, source_pos_disc, excitation_signal, courant_number, X, T, PML_damping=PML_damping)
    
    if ofname is not None:
        # Utworzenie okna wizualizacji
        fig = mlab.figure(size=(1000,1000))
    
    # Główna pętla obliczająca symulację FDTD
    for frame_number in range(1,sim_length):
        try:
            print('%3i/%3i\r'%(frame_number+1,sim_length), end='')
            
            # Obliczenie kroku symulacji i pogranie danych z GPU
            sim_obj.step()
            
            if frame_skip is not None:
                if (not (frame_number%frame_skip==0)) and (frame_number != 0):
                    continue
            
            p_distribution  = sim_obj.get_pressure_3D()
            
            if ofname is None:
                fig = mlab.figure(size=(1000,1000))
            
            # Wyświetlenie pola skalarnego reprezentującego
            # "środowisko" - czyli obiekty rozpraszające.
            environ_field   = mlab.pipeline.scalar_field(new_K)
            env_plot        = mlab.pipeline.volume(environ_field)
            
            # Wyświetlenie wizualizacji fali ciśnienia.
            pressure_wave   = mlab.pipeline.scalar_field(p_distribution, colormap='blue-red')
            vol_plot        = mlab.pipeline.volume(pressure_wave, vmin=0.01, vmax=np.max(p_distribution))
            
            # Pasek wartości na wizualizacji.
            lo_bound   = np.min([np.min(p_distribution),-vis_halfspread])
            hi_bound   = np.max([np.max(p_distribution),vis_halfspread])
            mlab.colorbar(orientation='vertical').data_range = (lo_bound, hi_bound)
            
            if display_time:
                mlab.title(f"time: {'%.2f'%(1000*frame_number*T)} [ms]",height=0.01,size=0.3)
            
            if ofname is not None:
                # Zapis klatki do folderu wyjściowego
                if frame_skip is not None:
                    effective_frame_number = frame_number//frame_skip
                else:
                    effective_frame_number = frame_number
            
                save_path = os.path.join(output_folder,f'{str(effective_frame_number).zfill(4)}.png')
                mlab.savefig(filename=save_path)
                # Wyczyszczenie klatki
                mlab.clf()
            else:
                mlab.show()
            
        except KeyboardInterrupt as e:
            print('Pętla renderująca została przerwana manualnie.\n')
            break
    
    # Dodanie nowej linii, aby nie zamazać ostatniej 
    # wartości licznika klatek symulacji.
    print()
    
    if ofname is not None:
        input('\n-------------------------\nZa chwilę klatki animacji zostaną zakodowane do postaci filmu.\nSprawdź klatki animacji w folderze i wciśnij ENTER by kontynuować\n')
    
        # Konwersja wyrenderowanych klatek do filmu.
        if frame_skip is not None:
            fps = int(60/frame_skip)
        else:
            fps = 60
    
        # Zakodowanie animacji do filmu w formacie .avi
        cmd = f"ffmpeg  -r {fps} -i _output_renders/%04d.png  -c:v libx264 -vf \"crop=trunc(iw/2)*2:trunc(ih/2)*2,fps={fps},format=yuv420p\" {ofname}"
        if encode_video:
            print(cmd)
            os.system(cmd)
        else:
            print(f'Jeśli chcesz zakodować wideo do finalnej postaci (filmiku), posłuż się komendą poniżej:\n\n{cmd}')
