DCASE2023 - Task 4 B - Baseline systems
-------------------------------------

Author:
**Irene Martin**, *Tampere University* 
[Email](mailto:irene.martinmorato@tuni.fi). 


Getting started
===============

1. Clone repository from [Github](https://github.com/marmoi/dcase2023_task4b_baseline).
2. Install requirements with command: `pip install -r requirements.txt`.
3. Extract features from the audio files **previously** downloaded `python feature_extraction.py`.
4. Run the task specific application with default settings `python task4b.py` or  `./task4b.py`


### Anaconda installation

To setup Anaconda environment for the system use following:

	conda create --name dcase-t4b python=3.6
	conda activate dcase-t4b
	conda install numpy
	conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
	pip install torchinfo
	pip install librosa
	pip install pandas
	pip install sklearn
	pip install sed_eval
    pip install dcase_util
    pip install sed_scores_eval
	


Introduction
============

This is the baseline system for the subtask of the Sound Event Detection task 4 of the Acoustic Scene Classification in Detection and Classification of Acoustic Scenes and Events 2023 (DCASE2023) challenge. The system is intended to provide a simple entry-level state-of-the-art approach that gives reasonable results. The baseline system is built on [dcase_util](https://github.com/DCASE-REPO/dcase_util) toolbox (>=version 0.2.16). 

Participants can build their own systems by extending the provided baseline system. 
The system is very simple, it does not handle dataset download, but a simple feature extraction
code is provided. The baseline system is a good starting point especially for the entry level researchers 
to familiarize themselves with the soft label scenario, numbers between 0 and 1. 

If participants plan to publish their code to the DCASE community after the challenge,
building their approach on the baseline system could potentially make their code more
accessible to the community. DCASE organizers strongly encourage participants to share
their code in any form after the challenge.

### Data preparation
     extract_features.py       # Code to extract features from the development files

Description
========

### Task 4 B - Sound Event Detection with Soft labels 

[MAESTRO Real - Multi-Annotator Estimated Strong Labels](https://zenodo.org/record/7244360) is used as development dataset for this task.

This task is a subtopic of the Sound Event Detection Task 4, which provides three 
kinds of data for training; weakly-labeled data (without timestamps), strongly-labeled data
(with timestamps) and unlabeled data. The target of the systems is to provide not only the 
event class but also the event time localization given that multiple events can be present in an audio recording

This task is concerned about another type of training data:
- Soft labels provide as a number between 0 and 1 that characterize the certainty of human annotators
 	for the sound at that specific time.
- Temporal resolution of the provided data is 1 second (due to the annotation procedure)
- Development data is provided with both soft (between 0 and 1) labels.
- Systems will be evaluated against hard labels



The task specific baseline system is implemented in file `model.py`.

#### System description

The system implements a convolutional recurren neural network (CRNN) based approach, 
with three CNN layers and one bi-directional gated recurrent unit (GRU) layer. As input, 
the model uses mel-band energies extracted using a hop length of 200 ms and 64 mel filter banks. 


#### Parameters

##### Acoustic features

- Analysis frame 400 ms (50% hop size)
- Mel-band energies (64 bands)

##### Neural network

- Input shape: sequence_length * 64 
- Architecture:
  - CNN layer #1
    - 2D Convolutional layer (filters: 128, kernel size: 3) + Batch normalization + ReLu activation
	- 2D max pooling (pool size: (1, 5)) + Dropout (rate: 20%)
  - CNN layer #2
    - 2D Convolutional layer (filters: 128, kernel size: 3) + Batch normalization + ReLu activation
    - 2D max pooling (pool size: (1, 2)) + Dropout (rate: 20%)
  - CNN layer #3
    - 2D Convolutional layer (filters: 32, kernel size: 3) + Batch normalization + ReLu activation
    - 2D max pooling (pool size: (1, 2)) + Dropout (rate: 20%)
  - Permute
  - Bidirectional #1
  - Dense layer #1
    - Dense layer (units: 64, activation: Linear )
    - Dropout (rate: 30%)
  - Dense layer #2
    - Dense layer (units: 32, activation: Linear )

- Learning (epochs: 150, batch size: 32, data shuffling between epochs)
  - Optimizer: Adam (learning rate: 0.001)
- Model selection:
  - Approximately 30% of the original training data is assigned to validation set
  - Model performance after each epoch is evaluated on the validation set, and best performing model is selected
  
**Network summary**

	Layer (type)                 Output Shape              Param #   
	=================================================================
	input_1 (InputLayer)        [(None, 1, 200, 64)]      0         
	_________________________________________________________________
	conv2d 		                (None, 128, 200, 64)      1280      
	_________________________________________________________________
	batch_normalization 		(None, 128, 200, 64)      256            
	_________________________________________________________________
	max_pooling2d 				(None, 128, 200, 12)      0         
	_________________________________________________________________
	dropout 		            (None, 128, 200, 12)      0         
	_________________________________________________________________
	conv2d_1 		            (None, 128, 200, 12)      147584    
	_________________________________________________________________
	batch_normalization_1       (None, 128, 200, 12)      256       
	_________________________________________________________________
	max_pooling2d_1       		(None, 128, 200, 6)       0         
	_________________________________________________________________
	dropout_1 		            (None, 128, 200, 6)       0         
	_________________________________________________________________
	conv2d_2           			(None, 128, 200, 6)       147584    
	_________________________________________________________________
	batch_normalization_2 		(None, 128, 200, 6)       256           
	_________________________________________________________________
	max_pooling2d_2  			(None, 128, 200, 3)       0         
	_________________________________________________________________
	dropout_2            		(None, 128, 200, 3)       0         
	_________________________________________________________________
	permute            			(None, 200, 128, 3)       0         
	_________________________________________________________________
	reshape_1           		(None, 200, 384)          0         
	_________________________________________________________________
	bidirectional 			 	(None, 200, 64)           80256           
	_________________________________________________________________
	Linear_1					(None, 200, 32)           2080            
	_________________________________________________________________
	Linear_2				    (None, 200, 17)           561         
	=================================================================

  
#### Results for development dataset

The cross-validation setup is used to evaluate the performance of the baseline system. Results are calculated using Pytorch in GPU mode (using Nvidia Tesla V100 GPU card). 
 
    
	| 	       | segment-based ER | segment-based F1 |	PSDS segment-based macro |	PSDS segment-based micro |
	|----------|------------------|------------------|---------------------------|---------------------------|
	| Baseline | 		0.48 	  |       70.1%   	 |			62.7% 	         |         	63.9%            |

                                                                                

**Note:** The reported system performance is not exactly reproducible due to varying setups. However, you should be able obtain very similar results.


Usage
=====

For running the CRNN model:
- `extract_features.py`, first extract mel-bands and normalize data
- `task4b.py`, DCASE2023 baseline for Task 1B


Code
====

The code is built on [dcase_util](https://github.com/DCASE-REPO/dcase_util) toolbox, see [manual for tutorials](https://dcase-repo.github.io/dcase_util/index.html). The machine learning part of the code is built on [Pytorch (v1.10.2)](https://pytorch.org/).

### File structure

      .
      ├── task4b.py                             # Baseline system for subtask B
      |
      ├── utils.py                              # Common functions shared between tasks
      ├── data_generator.py  					# File for the dataset
	  ├── extract_features.py					# Functions to extract mel-band features and normalize
	  ├── config.py								# Common parameters 
	  ├── evaluate.py							# Perform model evaluation, sed-eval segment-based
	  ├── model.py								# CRNN model implementation
      |
	  ├── development_folds						# Folder with the splits for 5-CV
	  |		- fold1_train.csv
	  |		- fold1_val.csv
	  |		- fold1_test.csv
	  |		- ...	
	  ├── metadata
	  |		- development_metadata.csv			# File duration information to calcualte sed-scores-eval
	  |		- gt_dev.csv						# Ground truth labels (hard-labels)
	  |
      ├── README.md                             # This file
      └── requirements.txt                      # External module dependencies

Changelog
=========

#### 1.0.0 / 2023-03-01



