from tensorflow.keras.layers import Conv2D,Input, Dense, MaxPooling2D, Flatten, Reshape, PReLU, ReLU, Concatenate, BatchNormalization, Dropout, Add
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from _imports.util_communication import *
from _imports.tmp_files          import *
import seaborn as sns
import pandas as pd
import shutil
import os
import copy as cp

def obtain_approximator_network_structure(input_sets, architecture_type):
    
    print(' ')
    print('tworzenie struktury sieci')

    input_shape = input_sets['X']['training'].shape[1::]
    input_layer = Input(shape=input_shape)
    
    def append_residual_block(in_x, num_chnls,intra_layers,norm_l2_val, dropout_val, batch_norm_momentum):
        x_A = Conv2D(num_chnls, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(in_x)
        x = Dropout(dropout_val)(x_A)
        for i in range(intra_layers):
            x = Conv2D(num_chnls, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
            x = Dropout(dropout_val)(x)
        x = Add()([x,x_A])
        return x
    
    def forward_pass_architecture_resnet(input_layer): 
        dropout_val    = 0.0
        norm_l2_val    = 0.0000
        res_block_size = 5
        batch_norm_momentum = 0.90
        
        x = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(input_layer)
        x = Dropout(0.1)(x)
        x = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Dropout(0.1)(x)
        
        x = MaxPooling2D((5,5))(x)
        xA = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(xA)
        x = Dropout(0.1)(x)
        x = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Dropout(0.1)(x)
        x = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Dropout(0.1)(x)
        x = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Dropout(0.1)(x)
        x = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Dropout(0.1)(x)
        x = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Dropout(0.1)(x)
        x = Add()([x,xA])
        
        xA = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(xA)
        x = Dropout(0.1)(x)
        x = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Dropout(0.1)(x)
        x = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Dropout(0.1)(x)
        x = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Dropout(0.1)(x)
        x = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Dropout(0.1)(x)
        x = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Dropout(0.1)(x)
        x = Add()([x,xA])
        
        xA = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(xA)
        x = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Conv2D(64, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Add()([x,xA])
        
        xA = Conv2D(128, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(xA)
        x = Conv2D(128, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Conv2D(128, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Conv2D(128, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Conv2D(128, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Conv2D(128, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Add()([x,xA])
        
        xA = Conv2D(256, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(xA)
        x = Conv2D(256, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Conv2D(256, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Conv2D(256, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Conv2D(256, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Conv2D(256, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Add()([x,xA])
        
        x = MaxPooling2D((2,2))(x)
        x = Conv2D(128, (2,2), activation=PReLU(), padding='same',kernel_regularizer=regularizers.l2(norm_l2_val))(x)
        x = BatchNormalization(momentum = batch_norm_momentum)(x)
        x = Flatten()(x)
        return x
        
    # Sieć neuronowa generująca odpowiedź w postaci
    # wartości skalarnej (nagrody)
    if architecture_type == 'scalar_reward':
        x      = forward_pass_architecture_resnet(input_layer)
        x      = Dense(32, activation=PReLU())(x)
        output = Dense(1, activation='linear')(x)
        approximator  = keras.Model(input_layer, output, name="approximator")
    
    # Jedna sieć neuronowa, z jednym "trzonem", która wylicza jednocześnie współczynniki
    # dyfuzji w płaszczyźnie xy i yz.
    elif architecture_type == 'matrix2d':
        x = forward_pass_architecture_resnet(input_layer)
        x             = Dense(4, activation=PReLU())(x)
        output        = Reshape((2,2))(x)
        
        approximator  = keras.Model(input_layer, output, name="approximator")
    
    # Dwie równoległę sieci neuronowe, z jednym "trzonem", które niezależnie obliczają
    # współczynniki dyfuzji w płaszczyźnie xy i yz.
    elif architecture_type == 'matrix_2d_separate_networks':
        norm_l2_val = 0.000
        output_xy = forward_pass_architecture_resnet(input_layer)
        x_xy = Dense(6, activation=PReLU(), kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(norm_l2_val))(output_xy)
        
        output_yz = forward_pass_architecture_resnet(input_layer)
        x_yz = Dense(6, activation=PReLU(), kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(norm_l2_val))(output_yz)
        
        x = Concatenate(axis=1)([x_xy, x_yz])
        output = Reshape((2,6))(x)
        
        approximator  = keras.Model(input_layer, output, name="approximator")
    
    else:
        raise RuntimeError(f'Wybrano zły typ architektury aproksymatora ({architecture_type})')
    
    return approximator

def draw_approximator_example(settings, input_sets_path):
    input_sets = np.load(input_sets_path, allow_pickle=True).item()
    data_storage_dir = settings['basic']['data_storage_dir']
    
    approximator_nn_type = settings['basic']['approximator_nn_type']
    approximator = obtain_approximator_network_structure(input_sets, approximator_nn_type)
    approximator.load_weights(os.path.join(data_storage_dir,'approximator_weights')).expect_partial()
    
    num_examples    = input_sets['X']['validation'].shape[0]
    example_idx     = np.random.randint(0,num_examples)
    input_example   = input_sets['X']['validation'][example_idx,:,:,:]
    input_example   = input_example[np.newaxis,:,:,:]
    
    _, Y = select_Y_set(approximator_nn_type, input_sets)
    
    expected_output  = Y[example_idx]
    predicted_output = approximator.predict(input_example)
    
    plt.figure()
    plt.plot(expected_output[0,:], label = 'expected 1')
    plt.plot(expected_output[1,:], label = 'expected 2')
    plt.plot(predicted_output[0,0,:], label = 'predicted 1')
    plt.plot(predicted_output[0,1,:], label = 'predicted 2')
    plt.grid()
    plt.legend()
    plt.ylim([0,1])

def draw_pred_and_val_boxplot(settings, input_sets_path):
    input_sets = np.load(input_sets_path, allow_pickle=True).item()
    data_storage_dir = settings['basic']['data_storage_dir']
    bands_f0         = settings['basic']['bands_f0']
    
    
    approximator_nn_type = settings['basic']['approximator_nn_type']
    approximator = obtain_approximator_network_structure(input_sets, approximator_nn_type)
    approximator.load_weights(os.path.join(data_storage_dir,'approximator_weights')).expect_partial()
    
    for set_type in ['training','validation']:
        print(f'generowany jest wykres dla zbioru: {set_type}')
        _, Y = select_Y_set(approximator_nn_type, input_sets)
        input_set        = input_sets['X'][set_type]
        predicted_output = approximator.predict(input_set)
        
        visualized_sets       = {set_type:Y,'predicted':predicted_output}
        long_format_vis_input = []
        for ex_num in range(Y.shape[0]):
            for f0_num, f0 in enumerate(bands_f0):
                for plane_num, plane_name in enumerate(['xy','yz']):
                    for set_num, set_name in enumerate(list(visualized_sets.keys())):
                        current_set = visualized_sets[set_name]
                        record_for_appending = {}
                        record_for_appending.update({'wąskop. wsp. dyfuzji':float(current_set[ex_num,plane_num,f0_num])})
                        record_for_appending.update({'f0':f0})
                        record_for_appending.update({'typ zbioru':set_name})
                        long_format_vis_input.append(record_for_appending)
        
        long_format_vis_input = pd.DataFrame(long_format_vis_input)
        
        plt.figure()
        sns.boxplot(data=long_format_vis_input, x='f0', y='wąskop. wsp. dyfuzji', hue='typ zbioru')
        plt.ylim([0,1])

def select_Y_set(approximator_nn_type, input_sets):
    if   approximator_nn_type in['matrix_2d_separate_networks','matrix2d']:
        Y_train = input_sets['Y_diff_coeffs']['training']
        Y_val   = input_sets['Y_diff_coeffs']['validation']
    # elif approximator_nn_type in['scalar_reward']:
        # Y_train = input_sets['Y_reward']['training']
        # Y_val   = input_sets['Y_reward']['validation']
    elif approximator_nn_type in['scalar_reward']:
        Y_train = input_sets['Y_wideband']['training']
        Y_val   = input_sets['Y_wideband']['validation']
    else:
        raise RuntimeError(f'Ustawiono zły typ sieci aproksymatora: {approximator_nn_type}')
    
    return Y_train, Y_val

def equalize_input_histogram(X_train, Y_train, num_chnls, n_bins = 20, max_examples_per_bin = 1000, non_boost_limit=50):
    input_data_df = pd.DataFrame(X_train)
    input_data_df = input_data_df.assign(Y_train=Y_train)
    input_data_df = input_data_df.sort_values(by='Y_train')

    y_min  = np.min(input_data_df['Y_train'])
    y_max  = np.max(input_data_df['Y_train'])
    print(f'zakres wartości Y: ({y_min},{y_max})')
    decimated_set = []
    bin_width = (y_max-y_min)/n_bins
    for i in range(n_bins):
        left_bnd  = y_min+i*bin_width
        right_bnd = y_min+(i+1)*bin_width
        
        left_mask  = input_data_df['Y_train'] >= left_bnd
        right_mask = input_data_df['Y_train'] <  right_bnd
        mask = np.logical_and(left_mask,right_mask)
        
        bin_contents = input_data_df.loc[mask,:]
        
        # Za dużo przykładów!
        if len(bin_contents) > max_examples_per_bin:
            bin_contents = bin_contents.sample(max_examples_per_bin)
        else:
            # Nic nie zrobimy, jeśli przykładów nie ma
            if len(bin_contents) <= non_boost_limit:
                continue
            # Za mało przykładów, oversampling!
            if len(bin_contents) != max_examples_per_bin:
                extra_examples = bin_contents.sample(max_examples_per_bin-len(bin_contents), replace=True)
                bin_contents = pd.concat([bin_contents, extra_examples])
                
        decimated_set.append(bin_contents)
    decimated_set = pd.concat(decimated_set)

    X_train = decimated_set.iloc[:,0:100*num_chnls].to_numpy()
    Y_train = decimated_set['Y_train'].to_numpy()
    
    return X_train, Y_train

def equalize_2d_input_histogram(X_train, Y_train):
    num_chnl = X_train.shape[3]
    X_train  = X_train.reshape(X_train.shape[0],100*num_chnl)
    
    n_bins = 40
    max_examples_per_bin = X_train.shape[0]//n_bins
    X_train, Y_train = equalize_input_histogram(X_train, Y_train, num_chnl, n_bins, max_examples_per_bin)
    
    X_train = X_train.reshape(X_train.shape[0],10,10,num_chnl)
    
    return X_train,Y_train

def train_approximator(settings, input_sets_path):
    
    approximator_nn_type = settings['basic']['approximator_nn_type']
    weights_training_backup_dir       = settings['basic']['weights_training_backup_dir']
    approximator_progress_logging_dir = settings['basic']['approximator_progress_logging_dir']
    data_storage_dir                  = settings['basic']['data_storage_dir']
    
    # Odczyt danych treningowych
    input_sets = np.load(input_sets_path, allow_pickle=True).item()
    X_train    = input_sets['X']['training']
    X_val      = input_sets['X']['validation']
    
    Y_train, Y_val = select_Y_set(approximator_nn_type, input_sets)
    
    # HACK - na razie przewidujemy tylko jeden współczynnik z puli wszystkich możliwych
    # Y_train, Y_val = select_Y_set('matrix2d', input_sets)
    
    
    # tworzenie struktury sieci
    approximator = obtain_approximator_network_structure(input_sets, approximator_nn_type)
    approximator.summary()
    
    # Y_val   = np.log(1-Y_val)
    # Y_val   = Y_val - np.mean(Y_val)
    # Y_train = np.log(1-Y_train)
    # Y_train = Y_train - np.mean(Y_train)
    
    Y_train_orig_vis = cp.copy(Y_train)
    
    # Hacks - do usunięcia gdy wyjaśni się struktura sieci
    if settings['basic']['approximator_nn_type'] == 'approximator_nn_type':
        X_train, Y_train = equalize_2d_input_histogram(X_train, Y_train)
    else:
        chosen_axes = [1,2]
        Y_train = Y_train[:,:,chosen_axes]
        Y_val   = Y_val  [:,:,chosen_axes]
    
    if False:
    # if True:
        plt.figure()
        plt.hist(Y_train_orig_vis, bins=100,label='before')
        plt.hist(Y_train, bins=100,label='after')
        plt.legend()
        plt.title('training')
        
        plt.show()
        exit()
    
    
    # DEBUG !!!! usunąć po sprawdzeniu jak radzą sobie drzewa decyzyjne!
    # expdat = {}
    # expdat.update({'X_train':X_train})
    # expdat.update({'X_val':X_val})
    # expdat.update({'Y_train':Y_train})
    # expdat.update({'Y_val':Y_val})
    # np.save('dectrees_export.npy', expdat)
    # exit()
    # DEBUG !!!! usunąć po sprawdzeniu jak radzą sobie drzewa decyzyjne!
    
    print()
    print('wymiary zbiorów danych:')
    print(f' X_train: {X_train.shape}')
    print(f' Y_train: {Y_train.shape}')
    print(f' X_val: {X_val.shape}')
    print(f' Y_val: {Y_val.shape}')
    print()
    
    # standardization
    if False:
        X_train = (X_train - np.mean(X_train))/(np.std(X_train))
        X_val   = (X_val - np.mean(X_val))/(np.std(X_val))
        
        Y_train = (Y_train - np.mean(Y_train))/(np.std(Y_train))
        Y_val   = (Y_val - np.mean(Y_val))/(np.std(Y_val))
    
    # Użytkownik może zdecydować, czy chce wczytać wagi z poprzedniego uruchomienia algorytmu, czy rozpocząć trening "od zera".
    train_metadata_fname = 'approximator_training_metadata.npy'
    train_metadata_path  = os.path.join(data_storage_dir,train_metadata_fname)
    initial_epoch_number = 0
    if ask_for_user_preference('Załadować ostatnie zachowane wagi sieci?'):
        # Wykorzystujemy blok try-except aby obsłużyć przypadek, gdy nie zapisano żadnych danych, które
        # mogłyby być odczytane.
        try:
            print('wagi zostaną przywrócone')
            approximator.load_weights(os.path.join(data_storage_dir,'approximator_weights'))
            train_metadata = np.load(train_metadata_path,allow_pickle=True).item()
            print(f"trening zostanie wznowniony od epoki {train_metadata['resuming_epoch_number']}, poprzedni trening zakończono ze stratą walidacyjną równą {train_metadata['termination_val_loss']}.")
            initial_epoch_number = train_metadata['resuming_epoch_number']
            
        except:
            RuntimeError('odczytywanie wag się nie powiodło - czy skrypt był już wywoływany i wydane zostało polecenie zapisu wag do późniejszego użytku?')
    else:
        print('trening zostanie rozpoczęty od zera')

    # Wartość współczynnika nauki może być wybrana przez użytkownika z listy lub może być podana jej dokładna wartość
    print()
    user_choice = ask_user_for_an_option_choice('Wybierz z listy poniżej współczynnik szybkości nauki (lr):', 'Numer wybranej opcji: ',[1e-3,1e-4,1e-5,'inna','pomiń trening'])
    skip_training = False
    if user_choice == 'pomiń trening':
        skip_training = True
    elif user_choice == 'inna':
        learning_rate = ask_user_for_a_float('Podaj wartość współczynnika nauki: ')
    else:
        learning_rate = user_choice

    if not skip_training:
        print('Wybrano wartość lr = '+str(learning_rate))
        print()
        
        opt_adam = optimizers.Adam(lr=learning_rate)
        opt_sgd  = optimizers.SGD(learning_rate=learning_rate, momentum=0.2, nesterov=True)
        
        opt = opt_adam
        # opt = opt_sgd
        
        # approximator.compile(optimizer=opt, loss='mean_squared_error')
        approximator.compile(optimizer=opt, loss='mean_absolute_error')
        # approximator.compile(optimizer=opt, loss='mean_squared_logarithmic_error')
        # approximator.compile(optimizer=opt, loss='huber_loss')
        
        log_dir=os.path.join(approximator_progress_logging_dir,create_timestamp())
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir,write_images=True,histogram_freq=True)

        # Własny callback, który monitoruje funkcję straty i zapisuje model w momencie osiągnięcia najlepszego
        # wyniku zanotowanego w całej historii treningu
        class BestSpecimenCallback(keras.callbacks.Callback):
            def __init__(self, saving_dir, file_name):
                self.best_loss        = None
                self.last_saving_path = None
                self.saving_dir       = saving_dir
                self.file_name        = file_name
            
            def on_epoch_end(self,epoch,logs=None):
                saving_indicator =''
                if (self.best_loss is None) or (logs["val_loss"]<self.best_loss):
                    self.best_loss = logs["val_loss"]
                    saving_path    = os.path.join(self.saving_dir,self.file_name)
                    self.model.save_weights(saving_path)
                    saving_indicator =' update!'
                    self.last_saving_path = saving_path
                logs.update({'avg_loss':logs["val_loss"]})
                logs.update({'best_loss':self.best_loss})
        
        run_timestamp                = create_timestamp()
        best_specimen_bkp_name       = 'best_specimen_'+run_timestamp
        best_specimen_saver_callback = BestSpecimenCallback(weights_training_backup_dir,best_specimen_bkp_name)
        
        # Wyświetlmy przypomnienie o tym, że trening można obserwować w TensorBoardzie
        print('\n\n')
        print('-----------------------------')
        print(' rozpoczęty zostanie proces treningu, pamiętaj, że możesz')
        print(' śledzić jego postępy w panelu narzędzia TensorBoard')
        print(' aby to zrobić, wpis w osobnym oknie konsoli poniższe polecenie:')
        print(' tensorboard --logdir="%s"'%(os.path.abspath(approximator_progress_logging_dir)))
        print('-----------------------------')
        print('\n\n')
        
        # Właściwa procedura treningowa
        try:
            while True:
                approximator.fit(
                    X_train, Y_train, 
                    validation_data = (X_val, Y_val), 
                    epochs = 10_000, initial_epoch=initial_epoch_number, batch_size=settings['basic']['batch_size'], 
                    callbacks=[tensorboard_callback,
                        best_specimen_saver_callback])
                
                print()
                print('ograniczenie epok osiągnięte, restart procedury treningowej')
                print()
                
        except KeyboardInterrupt:
            print('\n\ntrening przerwany ręcznie')
    
        print()
        current_best_path            = best_specimen_saver_callback.last_saving_path
        
        def move_result_file(weights_training_backup_dir, current_best_path):
            current_best_path_basename,_ = os.path.splitext(os.path.basename(current_best_path))
            for fname in os.listdir(weights_training_backup_dir):
                basename, ext = os.path.splitext(fname)
                
                if basename == current_best_path_basename:
                    old_path = os.path.join(weights_training_backup_dir, fname)
                    new_path = os.path.join(data_storage_dir, 'approximator_weights'+ext)
                    shutil.copy(old_path, new_path)
            
            runs_log_fname = 'approximator_training_runs_log.txt'
            runs_log_path = os.path.join(data_storage_dir,runs_log_fname)
            with open(runs_log_path,'a', encoding='utf-8') as f:
                row = ''
                row += f'entry date: {create_sparse_timestamp()}, '
                row += f'tensorboard_logging_path: {log_dir}, '
                row += f'learning_rate: {learning_rate}, '
                row += f"batch_size: {settings['basic']['batch_size']}"
                row += f'\n'
                f.write(row)
            
            val_loss_hist = approximator.history.history['val_loss']
            resuming_epoch_number =len(val_loss_hist)+initial_epoch_number
            
            train_metadata = {}
            train_metadata.update({'resuming_epoch_number':resuming_epoch_number})
            train_metadata.update({'termination_val_loss':val_loss_hist[-1]})
            np.save(train_metadata_path,train_metadata)
        
        last_specimen_bkp_name = 'last_training_weights'+run_timestamp
        approximator.save_weights(os.path.join(weights_training_backup_dir,last_specimen_bkp_name))
        
        if current_best_path is not None:
            if ask_for_user_preference('Czy przenieść ostatni najlepszy wynik do folderu roboczego?'):
                move_result_file(weights_training_backup_dir, current_best_path)
            else:
                if ask_for_user_preference('Czy przenieść ostatni uzyskany wynik do folderu roboczego?'):
                    move_result_file(weights_training_backup_dir, last_specimen_bkp_name)
                    
            
    
    if ask_for_user_preference('Czy zwizualizować wyniki regresji?'):
        decim_factor_val = 1
        decim_factor_trn = 10
        if settings['basic']['augment_training_set']:
            decim_factor_val = 10
            decim_factor_trn = 1000
        predicted = approximator.predict(X_val)
        plt.figure()
        if len(Y_val.shape)==3:
            for i in range(Y_val.shape[1]):
                for j in range(Y_val.shape[2]):
                    plt.scatter(Y_val[::decim_factor_val,i,j],predicted[::decim_factor_val,i,j])
        else:
            plt.scatter(Y_val[::decim_factor_val],predicted[::decim_factor_val])
        plt.title('regression on validation set')
        plt.xlabel('Y_val')
        plt.ylabel('predicted')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.grid()
        
        predicted = approximator.predict(X_train)
        plt.figure()
        if len(Y_train.shape)==3:
            for i in range(Y_train.shape[1]):
                for j in range(Y_train.shape[2]):
                    plt.scatter(Y_train[::decim_factor_val,i,j],predicted[::decim_factor_val,i,j])
        else:
            plt.scatter(Y_train[::decim_factor_val],predicted[::decim_factor_val])
        plt.title('regression on training set')
        plt.xlabel('Y_train')
        plt.ylabel('predicted')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.grid()
        
        plt.show()
