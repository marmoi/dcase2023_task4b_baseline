import os
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from evaluate import *
from model import *
from utils import split_in_seqs, create_folder, move_data_to_device
import config 
import sed_eval
import pandas as pd
from data_generator import maestroDataset


def load_merged_data(_feat_folder, _lab_folder,  _fold=None):
    # Load features (mbe)
    feat_file_fold = os.path.join(_feat_folder, 'merged_mbe_fold{}.npz'.format( _fold))
    dmp = np.load(feat_file_fold)

    _X_train, _X_val = dmp['arr_0'], dmp['arr_1']

    # Load the corresponding labels
    lab_file_fold = os.path.join(_lab_folder, 'merged_lab_soft_fold{}.npz'.format(_fold))
    dmp = np.load(lab_file_fold)
    _Y_train, _Y_val = dmp['arr_0'], dmp['arr_1']

    return _X_train, _Y_train, _X_val, _Y_val


def preprocess_data(_X, _Y, _X_val, _Y_val, _seq_len):
    # split into sequences
    _X = split_in_seqs(_X, _seq_len)
    _Y = split_in_seqs(_Y, _seq_len)

    _X_val = split_in_seqs(_X_val, _seq_len)
    _Y_val = split_in_seqs(_Y_val, _seq_len)

    return _X, _Y, _X_val, _Y_val


def train():
    # Arguments & parameters
    stop_iteration = 150
    learning_rate = 1e-3 
    patience = int(0.6*stop_iteration)
    holdout_fold = np.arange(1, 6)
    seq_len = 200
    batch_size = 32
    
    # CRNN model definition   
    cnn_filters = 128       # Number of filters in the CNN
    rnn_hid = 32            # Number of RNN nodes.  Length of rnn_nb =  number of RNN layers
    dropout_rate = 0.2      # Dropout after each layer


    device = 'cuda' if (torch.cuda.is_available()) else 'cpu'

    # Add variables to save the test loss
    avg_er = list(); avg_f1 = list()
    print(f'Learning rate {learning_rate} - sequence length {seq_len} - batch_size {batch_size}')

    # For evaluating the model, only hard labels will be considered (11 classes)
    segment_based_metrics_all_folds = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=config.labels_hard,
        time_resolution=1.0
    )


    # Create output folders
    output_model = 'model_crnn'
    create_folder(output_model)

    output_folder = 'dev_txt_scores'
    create_folder(output_folder)

    for fold in holdout_fold:

        # Load features and labels
        X, Y, X_val, Y_val = load_merged_data('development/features', 'development/soft_labels', fold)
        X, Y, X_val, Y_val = preprocess_data(X, Y, X_val, Y_val, seq_len)
        
        train_dataset = maestroDataset(X, Y)
        validate_dataset = maestroDataset(X_val, Y_val)

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                    num_workers=1, pin_memory=True)

        validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=True,
                                                    num_workers=1, pin_memory=True)

        # Prepare model
        modelcrnn = my_CRNN(config.classes_num_soft, cnn_filters, rnn_hid, dropout_rate)
        
        if 'cuda' in device:
            modelcrnn.to(device)
        print('\nCreate model:')
        if fold == 1:
            import nessi
            nessi.get_model_size(modelcrnn, 'torch' ,input_size=(1,seq_len, 64))

        # Optimizer
        optimizer = optim.Adam(modelcrnn.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=False)

        best_epoch = 0; pat_cnt = 0; pat_learn_rate = 0; best_loss = 99999
        tr_loss, val_F1, val_ER = [0] * stop_iteration, [0] * stop_iteration, [0] * stop_iteration

        # Train on mini batches
        tr_batch_loss = list()       
        for epoch in range(stop_iteration):
            
            modelcrnn.train()
            # TRAIN
            for (batch_data, batch_target) in train_loader:
                # Zero gradients for every batch
                optimizer.zero_grad()

                batch_output = modelcrnn(move_data_to_device(batch_data, device))

                # Calculate loss
                loss = clip_mse(batch_output, move_data_to_device(batch_target,device))
                    
                tr_batch_loss.append(loss.item())

                # Backpropagation
                loss.backward()
                optimizer.step()

            tr_loss[epoch] = np.mean(tr_batch_loss)
        
            # VALIDATE
            modelcrnn.eval()

            with torch.no_grad():

                segment_based_metrics_batch = sed_eval.sound_event.SegmentBasedMetrics(
                    event_label_list=config.labels_hard,
                    time_resolution=1.0
                )
                
                running_loss = 0.0
                for (batch_data, batch_target) in validate_loader:
                    

                    batch_output = modelcrnn(move_data_to_device(batch_data, device))

                    loss = clip_mse(batch_output, move_data_to_device(batch_target,device))

                    segment_based_metrics_batch = metric_perbatch(segment_based_metrics_batch,
                                                                  batch_output.reshape(-1, len(config.labels_soft)).detach().cpu().numpy(),
                                                                  batch_target.reshape(-1, len(config.labels_soft)).numpy())

                    running_loss += loss

                avg_vloss = running_loss /len(validate_loader)
                
                batch_segment_based_metrics_ER = segment_based_metrics_batch.overall_error_rate()
                batch_segment_based_metrics_f1 = segment_based_metrics_batch.overall_f_measure()
                val_F1[epoch] = batch_segment_based_metrics_f1['f_measure']
                val_ER[epoch] = batch_segment_based_metrics_ER['error_rate']
                
                # Check if during the epochs the ER does not improve
                if avg_vloss < best_loss:
                    best_model = modelcrnn
                    best_epoch = epoch
                    best_loss = avg_vloss
                    pat_cnt = 0
                    pat_learn_rate = 0
                    output = segment_based_metrics_batch.result_report_class_wise()

                    print(output)
                    torch.save(best_model.state_dict(), f'{output_model}/best_fold{fold}.bin')

            pat_cnt += 1
            pat_learn_rate += 1

            if pat_learn_rate > int(0.3 * stop_iteration):
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']/10
                    pat_learn_rate = 0
                    print(f'\tDecreasing learning rate to:{g["lr"]}')

            print(f'Epoch: {epoch} - Train loss: {round(tr_loss[epoch],3)} - Val loss: {round(avg_vloss.item(),3)}'
                  f' - val F1 {round(val_F1[epoch]*100,2)} - val ER {round(val_ER[epoch],3)}'
                  f' - best epoch {best_epoch} F1 {round(val_F1[best_epoch]*100,2)}')

            segment_based_metrics_batch.reset()
            # Stop learning
            if (epoch == stop_iteration) or (pat_cnt > patience):
                break

        # TEST
        test_files = pd.read_csv('development_folds/fold{}_test.csv'.format(fold))['filename'].tolist()                                            
        
        segment_based_metrics_test = sed_eval.sound_event.SegmentBasedMetrics(
            event_label_list=config.labels_hard,
            time_resolution=1.0
        )

        
        modelcrnn.load_state_dict(torch.load(f'{output_model}/best_fold{fold}.bin', map_location=device))
        modelcrnn.eval()
        
        with torch.no_grad():
            nbatch = 0
            for file in test_files:
                # Load the corresponding audio file
                audio_name = file.split('/')[-1]
                batch_data = np.load(f'development/features/test_{audio_name}_fold{fold}.npz')
                data = torch.Tensor(batch_data['arr_0'])
                batch_target = np.load(f'development/soft_labels/lab_soft_{audio_name}_fold{fold}.npz')
                target = batch_target['arr_0']
                
                # Feed into the model
                batch_output = modelcrnn(data[None,:,:].to(device))
                framewise_output = batch_output.squeeze().detach().cpu().numpy()
                
                # output for each file
                eval_meta(output_folder, audio_name, framewise_output)

                # Append to evaluate the whole test fold at once
                if nbatch == 0:
                    fold_target = target
                    fold_output = framewise_output
                else:
                    fold_target = np.append(fold_target, target, axis=0)
                    fold_output = np.append(fold_output, framewise_output, axis=0)

                nbatch += 1
            

            reference = process_event(config.labels_soft, fold_target.T, config.posterior_thresh,
                                        config.hop_size / config.sample_rate)

            results = process_event(config.labels_soft, fold_output.T, config.posterior_thresh,
                                            config.hop_size / config.sample_rate)

            segment_based_metrics_test.evaluate(
                reference_event_list=reference,
                estimated_event_list=results
            )

            # Save data for all the folds
            segment_based_metrics_all_folds.evaluate(
                reference_event_list=reference,
                estimated_event_list=results
            )


            output = segment_based_metrics_test.result_report_class_wise()
            print(output)
            overall_segment_based_metrics_ER = segment_based_metrics_test.overall_error_rate()
            overall_segment_based_metrics_f1 = segment_based_metrics_test.overall_f_measure()
            f1_overall_1sec_list = overall_segment_based_metrics_f1['f_measure']
            er_overall_1sec_list = overall_segment_based_metrics_ER['error_rate']
            segment_based_metrics_test.reset()

            print(f'FOLD {fold} - ER: {round(er_overall_1sec_list,5)} F1: {round(f1_overall_1sec_list,5)} \n')
            print('*-----------------------------------------------------------------------*')
            avg_er.append(er_overall_1sec_list); avg_f1.append(f1_overall_1sec_list)

            # empty GPU cache
            torch.cuda.empty_cache()

    # End folds
    print('*-----------------------------------------------------------------------*')
    print('\nResult for ALL FOLDS: \n\tER: {} \n\tF1: {} '.format(avg_er,  avg_f1))

    output = segment_based_metrics_all_folds.result_report_class_wise()
    print(output)
    overall_segment_based_metrics_ER = segment_based_metrics_all_folds.overall_error_rate()
    overall_segment_based_metrics_f1 = segment_based_metrics_all_folds.overall_f_measure()
    f1_overall_1sec_list = overall_segment_based_metrics_f1['f_measure']
    er_overall_1sec_list = overall_segment_based_metrics_ER['error_rate']
    print(f'\nMicro segment based metrics - ER: {round(er_overall_1sec_list,3)} F1: {round(f1_overall_1sec_list*100,2)} ')
    class_wise_metrics = segment_based_metrics_all_folds.results_class_wise_metrics()
    macroFs = []
    for c in class_wise_metrics:
        macroFs.append(class_wise_metrics[c]["f_measure"]["f_measure"])
    print(f'\nMacro segment based F1: {round((sum(np.nan_to_num(macroFs))/config.classes_num_hard)*100,2)} ')
    print('\n')
    path_groundtruth = 'metadata/gt_dev.csv'
    # Calculate PSDS SED metrics
    get_PSDS(path_groundtruth, output_folder)


# python train_soft.py 
if __name__ == '__main__':

    train()
