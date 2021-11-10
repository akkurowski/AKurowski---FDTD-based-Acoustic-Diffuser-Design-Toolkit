# ---------------------------------------------------- #
# Pakiet procedur do generowania geometrii obiektów
# symulowanych metodą FDTD
# author: Adam Kurowski
# date:   25.01.2020
# e-mail: akkurowski@gmail.com
# ---------------------------------------------------- #

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import mayavi.mlab
import numba as nb
import time

# ------------------------------------------------
# Funkcje pomocnicze do generowania kształtów
# ------------------------------------------------

# Funkcja przetwarzająca "formę" obiektu w postaci macierzy 
# gdzie 1 to woksel zajęty przez obiekt, a 0 to woksel
# niezajęty przez obiekt to formatu wymaganego przez FDTD
# (5 - powierzchnia, 4 - krawędź, 3 - wierzchołek)
# 
# Woksele formy (mold) MUSZĄ MIEĆ WARTOŚĆ 1!
# 
# Dodatkowo wszelkie prostopadłościenne kształty w 
# domenie obliczeniowej NIE MOGĄ MIEĆ GRUBOŚCI 1 WOKSELA
# bo wtedy funkcja bake nie działa poprawnie 
# (musi być grubość przynajmniej 2 wokseli)

def bake(shape_mold):
    # Różniczkowanie dwukierunkowe - nakładane na jeden wymiar
    # w celu detekcji narożników i krawędzi
    def bidirect_differencing(mtx, axis, padding='prepend'):
        output  = np.zeros_like(mtx)
        
        if padding == 'prepend':
            kwargs = {'prepend':0}
        elif padding == 'append':
            kwargs = {'append':0}
        else:
            raise RuntimeError(f'bad value specified for padding keyword argument: {padding}')
        
        mtx_rev = np.flip(mtx,axis=axis)
        output += np.diff(mtx,axis=axis, **kwargs)
        output += np.flip(np.diff(mtx_rev,axis=axis, **kwargs),axis=axis)
        
        return output

    # Nałożenie różniczkowania dwukierunkowego na całą
    # trójwymiarową przestrzeń - uzyskany wynik pozwala
    # np. na detekcję powierzchni, krawędzi, czy wierzchołków.
    def unidirectional_differencing(shape_mold):
        output  = np.zeros_like(shape_mold)
        output += bidirect_differencing(shape_mold, 0)
        output += bidirect_differencing(shape_mold, 1)
        output += bidirect_differencing(shape_mold, 2)
        return output

    # Aplikacja różniczkowania bezkierunkowego i ograniczenie
    # rezultatów tylko do obszaru zajmowanego przez wzorzec
    # geometrii
    diff_stencil = unidirectional_differencing(shape_mold)
    diff_stencil[np.where(shape_mold==0)] = 0
    
    # Dopasowanie wygenerowanej geometrii do numerów 
    # odpowiadających warunkom brzegowym (5 - płaszczyzna, 
    # 4 - krawędź, 3 - róg, 0 - wnętrze obiektu, 
    # 6 - medium propagacyjne)
    baked_object = np.zeros_like(diff_stencil)
    baked_object[diff_stencil==1] = 5
    baked_object[diff_stencil==2] = 4
    baked_object[diff_stencil==3] = 3
    baked_object[shape_mold  ==0] = 6
    
    return baked_object

# Wiele operacji bazuje na osadzaniu "formy" (mold) 
# złożonej z wielu prostopadłościanów - stąd bazowa funkcja
# do łatwego osadzania ich i nadawania wokselom wchodzącym
# w ich skład wybranej wartości (value), zwykle - 1
# (wymagane przez funkcję bake)
def embed_cuboid(K, x_pos, y_pos, z_pos, width, length, height, value=1):
    K[x_pos:x_pos+width, y_pos:y_pos+length, z_pos:z_pos+height] = value
    return K

# ------------------------------------------------
# Generowanie kształtów
# ------------------------------------------------

# Znaczenie osi w przestrzeni symulacji:
#   z - wysokosc domeny obliczeniowej w symulacji FDTD
#   y - glebokosc dyfuzora w domenie
#   x - szerokosc dyfuzora (kier od góry do dołu przy 
#   rzucie z gory - jak na wizualizacjach)

# Obiekt domeny obliczeniowej spełniającej założenia 
# równania FDTD z macierzą K.
def make_computational_domain(x_dim,y_dim,z_dim):
    # Alokacja miejsca na domenę obliczeniową.
    plate_obj = np.zeros((x_dim, y_dim, z_dim))
    
    # Granica domeny obliczeniowej ograniczona jest 
    # warunkiem typu "powierzchnia".
    plate_obj = embed_cuboid(plate_obj, 1, 1, 1, x_dim-2, y_dim-2, z_dim-2, value=1)
    
    # Konwersja geometrii na format docelowy.
    plate_obj = bake(plate_obj)
    
    # Konieczne jest pozostawienie warstwy zer poza 
    # powierzchnią domeny, inaczej równanie propagacji 
    # stanie się niestabilne, stąd podmiana:
    plate_obj[plate_obj==6] = 0
    
    # Dodatkowo, środek domeny wypełniany jest wartością 
    # oznaczającą medium propagacyjne.
    plate_obj = embed_cuboid(plate_obj, 2, 2, 2, x_dim-4, y_dim-4, z_dim-4, value=6)
    
    return plate_obj

# Osadzenie kształtu (np. dyfuzora) w domenie obliczeniowej
# w zadanej pozycji w przestrzeni 3D.
def embed_shape_into_domain(computational_domain, beta, shape_obj, admittance, x_pos=0, y_pos=0, z_pos=0):
    
    computational_domain[
        x_pos:(x_pos+shape_obj.shape[0]),
        y_pos:(y_pos+shape_obj.shape[1]),
        z_pos:(z_pos+shape_obj.shape[2])] = shape_obj[:,:,:]
    
    for vox_type in [5,4,3,0]:
        mask_args = list(np.where(shape_obj==vox_type))
        mask_args[0]    += x_pos
        mask_args[1]    += y_pos
        mask_args[2]    += z_pos
        mask_args        = tuple(mask_args)
        beta[mask_args]  = admittance
    
    return computational_domain, beta

# Generator kształtu dyfuzora Skyline (tm) z patternem 1D
def get_1D_Skyline_diffuser(pattern=[0,1,0,0,1,2,1,1,2,1,0,0,1,0,0,1,0,0,1,2,1,1,2,1,0,0,1,0],segment_height=15,segment_width=6,diffuser_height = 120):

    # Wejście zawsze powinno być typu numpy.array
    if type(pattern) == list:
        pattern = np.array(pattern)
    
    # Obliczenie informacji o wymiarach segmentów i
    # całego dyfuzora
    num_segments    = len(pattern)
    segment_heights = (pattern+1)*segment_height
    segment_heights = segment_heights.tolist()
    diffuser_width = num_segments*segment_width
    diffuser_elem_height = num_segments*segment_width
    
    # Przygotowanie formy i wygenerowanie prostopadłościanów
    # które im odpowiadają
    shape_mold = np.zeros((diffuser_width,diffuser_elem_height,diffuser_height))
    for i, height in enumerate(segment_heights):
        shape_mold = embed_cuboid(shape_mold,i*segment_width,0,0, segment_width, height, diffuser_height)
    
    # Konwersja formy do końcowej postaci kształtu dyfuzora.
    baked_object = bake(shape_mold)
    
    # Usunięcie wierzchołków i krawędzi (dla stabilności)
    baked_object[baked_object==3] = 6
    baked_object[baked_object==4] = 6
    
    return baked_object

# Obiekt klasycznego dyfuzora 2D Schroedera
def generate_2D_Skyline_diffuser(pattern = [[1,0,2,1],[0,0,3,0],[2,3,0,1]],element_size=10,element_seg_depth=10):
    # Wejście zawsze powinno być typu numpy.array
    if type(pattern) == list:
        pattern = np.array(pattern)
    
    # Wymiary dyfuzora
    diffuser_height = pattern.shape[0]*element_size
    diffuser_width  = pattern.shape[1]*element_size
    
    # Dostosowanie układu wgłębień do pętli generującej
    # kształt dyfuzora (konieczne, bo zakładamy, że punkt 
    # 0,0 jest na dole po lewej stronie dyfuzora).
    flipped_pattern = np.flip(pattern, axis=0)
    flipped_pattern = np.flip(flipped_pattern, axis=1)
    flipped_pattern = flipped_pattern + 1
    
    # Przygotowanie macierzy w której osadzona zostanie
    # geometria dyfuzora.
    diffuser_depth = np.max(flipped_pattern)*element_seg_depth
    diffuser_obj   = np.zeros((diffuser_width,diffuser_depth,diffuser_height))
    
    # Pętla wycinająca wgłębienia dyfuzora.
    for col_idx in range(flipped_pattern.shape[0]):
        for row_idx in range(flipped_pattern.shape[1]):
            
            # Koordynaty wgłębienia w przestrzeni 3D
            element_pos_x = element_size*row_idx
            element_pos_z = element_size*col_idx
            element_pos_y = 0
            
            # Wymiary wgłębienia 
            element_size_x = element_size
            element_size_z = element_size
            element_size_y = flipped_pattern[col_idx,row_idx]*element_seg_depth
            
            # Wycięcie wgłębienia w bryle dyfuzora
            diffuser_obj = embed_cuboid(diffuser_obj,
                element_pos_x,element_pos_y,element_pos_z,
                element_size_x,element_size_y,element_size_z,
                value=1)
    
    # Konwersja kształtu do formatu docelowego.
    diffuser_obj = bake(diffuser_obj)
    
    # Usunięcie wierzchołków i krawędzi (dla stabilności)
    diffuser_obj[diffuser_obj==3] = 6
    diffuser_obj[diffuser_obj==4] = 6
    
    return diffuser_obj

# Obiekt płaskiej płyty - potrzebny np. do pomiaru
# współczynnika rozproszenia i dyfuzji
def generate_plate(x_dim=50,y_dim=50,z_dim=100):
    # Wygenerowanie formy płyty
    plate_obj = np.zeros((x_dim, y_dim, z_dim))
    plate_obj = embed_cuboid(plate_obj, 0, 0, 0, x_dim, y_dim, z_dim, value=1)
    
    # Konwersja matrycy do formatu z rozróżnieniem krawędzi i wierzchołków.
    plate_obj = bake(plate_obj)
    
    # Usunięcie wierzchołków i krawędzi (dla stabilności)
    plate_obj[plate_obj==3] = 6
    plate_obj[plate_obj==4] = 6
    
    return plate_obj

# Obiekt klasycznego dyfuzora 2D Schroedera
def generate_2D_Schroedder_diffuser(pattern = [[1,0,2,1],[0,0,3,0],[2,3,0,1]],well_size=10,well_seg_depth=10,diffuser_depth=40):
    
    # Wejście zawsze powinno być typu numpy.array
    if type(pattern) == list:
        pattern = np.array(pattern)
    
    # Grubość ścianek dyfuzora - aby procedura bake
    # działała poprawnie nie może ona być mniejsza niż 3.
    # Powinna być ona możliwie mała więc ostatecznie
    # równa jest ona 3.
    WELL_WALL_WIDTH = 3
    
    # Przeskok pomiędzy analogicznymi punktami każdego
    # wgłębienia (well).
    well_increment  = (WELL_WALL_WIDTH+well_size)
    
    # Wymiary dyfuzora
    diffuser_height = pattern.shape[0]*well_increment + WELL_WALL_WIDTH
    diffuser_width  = pattern.shape[1]*well_increment + WELL_WALL_WIDTH
    
    # Dostosowanie układu wgłębień do pętli generującej
    # kształt dyfuzora (konieczne, bo zakładamy, że punkt 
    # 0,0 jest na dole po lewej stronie dyfuzora).
    flipped_pattern = np.flip(pattern, axis=0)
    flipped_pattern = np.flip(flipped_pattern, axis=1)
    
    # Przygotowanie macierzy w której osadzona zostanie
    # geometria dyfuzora.
    diffuser_obj = np.ones((diffuser_width,diffuser_depth,diffuser_height))
    
    # Pętla wycinająca wgłębienia dyfuzora.
    for col_idx in range(flipped_pattern.shape[0]):
        for row_idx in range(flipped_pattern.shape[1]):
            
            # Koordynaty wgłębienia w przestrzeni 3D
            well_pos_x = WELL_WALL_WIDTH+well_increment*row_idx
            well_pos_z = WELL_WALL_WIDTH+well_increment*col_idx
            # kierunek y musi być modulowany, by modulować
            # głębokość wgłębień
            well_pos_y = WELL_WALL_WIDTH + flipped_pattern[col_idx,row_idx]*well_seg_depth
            
            # Wymiary wgłębienia 
            well_size_x = well_size
            well_size_z = well_size
            # kierunek y musi być modulowany, by modulować
            # głębokość wgłębień
            well_size_y = diffuser_depth-WELL_WALL_WIDTH-flipped_pattern[col_idx,row_idx]*well_seg_depth
            
            # Wycięcie wgłębienia w bryle dyfuzora
            diffuser_obj = embed_cuboid(diffuser_obj,
                well_pos_x,well_pos_y,well_pos_z,
                well_size_x,well_size_y,well_size_z,
                value=0)
    
    # Konwersja kształtu do formatu docelowego.
    diffuser_obj = bake(diffuser_obj)
    
    # Usunięcie wierzchołków i krawędzi (dla stabilności)
    diffuser_obj[diffuser_obj==3] = 6
    diffuser_obj[diffuser_obj==4] = 6
    
    return diffuser_obj

# ------------------------------------------------
# Benerowanie macierzy beta
# ------------------------------------------------

# Nadanie wartości parametru admitancji beta warunkom brzegowym 
# domeny obliczeniowej.
# Ważne, by funkcję wywołać na ŚWIEŻO UTWORZONEJ MACIERZY 
# w przypadku, w którym konieczne jest przypisanie wartości 
# do ścian domeny obliczeniowej.
# (K musi być bez osadzonych innych obiektów)
def make_beta(K, volume_init_val=0, all_walls_init=None,
    beta_wall_left=None,   beta_wall_right=None,
    beta_wall_bottom=None, beta_wall_top=None, 
    beta_wall_down=None,   beta_wall_up=None):
    
    # Inicjalizacja macierzy impedancji
    beta          = np.ones_like(K)*volume_init_val
    
    if all_walls_init is None:
        walls_init_values = np.zeros(6)
    else:
        walls_init_values = np.ones(6)*all_walls_init
    
    for idx, value in enumerate([beta_wall_top,beta_wall_bottom,beta_wall_left,
    beta_wall_right,beta_wall_up,beta_wall_down]):
        if value is not None:
            walls_init_values[idx] = value
    
    beta[0:2,:,:]   = walls_init_values[0]
    beta[:,0:2,:]   = walls_init_values[2]
    beta[:,:,0:2]   = walls_init_values[4]
    beta[-2:beta.shape[0],:,:] = walls_init_values[1]
    beta[:,-2:beta.shape[1],:] = walls_init_values[3]
    beta[:,:,-2:beta.shape[2]] = walls_init_values[5]
    
    return beta

def make_BK(K,lam,beta):
    BK     = (6-K)*lam*beta/2
    return BK

# ------------------------------------------------
# Wizualizacja geometrii
# ------------------------------------------------

# Trójwymiarowy podgląd wygenerowanych geometrii dyfozorów
# (za pomocą biblioteki Matplotlib) 
def plot_with_color(K,ax, node_type_number, color):
    colors = np.empty(K.shape, dtype='object')
    voxels = np.zeros_like(K)
    colors[K==node_type_number] = color
    voxels[K==node_type_number] = 1
    ax.voxels(voxels, facecolors=colors, edgecolor='k', alpha = 0.5)

# Trójwymiarowy podgląd wygenerowanych geometrii dyfozorów
# (za pomocą biblioteki Mayavi) 
def show_shape(K, opacity = 1, mode='voxel', show_result=True):
    
    if mode == 'voxel':
        objects = []
        objects.append({'K_number':5,'color':(0.4, 0.4, 1)}) # powierzchnie
        objects.append({'K_number':4,'color':(0.4, 1, 0.4)}) # krawędzie
        objects.append({'K_number':3,'color':(1, 0.4, 0.4)}) # wierzchołki
        
        # Woksele medium propagacyjnego (6) 
        # nie są wizualizowane.
        
        for obj_def in objects:
            xx, yy, zz = np.where(K == obj_def['K_number']) # powierzchnie
            mayavi.mlab.points3d(xx, yy, zz, mode="cube", color=obj_def['color'], scale_factor=1, opacity=opacity, line_width=5)
    
    elif mode=='scalar_field':
        scalar_field = mayavi.mlab.pipeline.scalar_field(K,opacity=opacity)
        env_plot     = mayavi.mlab.pipeline.volume(scalar_field)
        mayavi.mlab.colorbar(orientation='vertical')
    
    elif mode=='scalar_cut_plane':
        scalar_field = mayavi.mlab.pipeline.scalar_field(K,opacity=opacity)
        mayavi.tools.modules.scalar_cut_plane(scalar_field, plane_orientation='x_axes')
        mayavi.tools.modules.scalar_cut_plane(scalar_field, plane_orientation='y_axes')
        mayavi.tools.modules.scalar_cut_plane(scalar_field, plane_orientation='z_axes')
        mayavi.mlab.colorbar(orientation='vertical')
    
    else:
        raise RuntimeError('Wybrano niewłaściwą opcję wizualizacji.')
    
    if show_result:
        # Pokazanie wyniku
        mayavi.mlab.show()