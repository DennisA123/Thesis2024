# Master Thesis Artificial Intelligence: Assessing and Addressing Gender Bias in Large Language Models

This is the repository for my Master Thesis at the University of Amsterdam.

## Introduction

This repository contains the code to run the experiments discussed in the thesis. The code for the bias assessment methods is in the folder `Assessment/`. The code for the two variants of the disentanglement model is in the folders `GRU_DM/` and `Transformer_DM/`. The code for the debiasing method is in `Debiasing`.

## Requirements

- **Python version:** 3.11.4
- **Torch and CUDA version:** 2.2.0+cu121

The file `requirements.txt` contains the necessary packages. All experiments were conducted using Ubuntu.

## Bias Assessment Experiments

### Folder Structure:

`Assessment/`  
├── `Data/`  
├── `methods.py`  
├── `run_seat.py`  
├── `run_csps.py`  
├── `run_disco.py`  
├── `run_sentiment.py`  

- `Data/`: Contains the datasets used for the bias assessment experiments.  
- `methods.py`: Contains the implementations of the assessment methods.  
- `run_<method>.ipynb`: Jupyter notebooks containing the executions of the bias assessment methods across the model and dataset variations.  

## Disentanglement Model Experiments  

`T_DM/`  
├── `Data/`  
├── `main_dem.py`  
├── `dem.py`  
├── `train_dem.py`  
├── `eval_dem.py`  
├── `utils.py`  

- `Data/`: Contains the datasets used for generating the BoW vectors, training the T-DM and evaluating the T-DM.  
- `main_dem.py`:  Pre-processes the BoW vector vocab data, train data and test data. Main file for running the training and evaluation experiments of the T-DM.  
- `dem.py`:  Contains the T-DM model, and the two types of feedforward networks/classifiers (D3, D4, D5 and D6).  
- `train_dem.py`:  Contains the training code of the T-DM.  
- `eval_dem.py`:  Contains the evaluation code of the T-DM.  
- `utils.py`:  Contains all the helper functions.  

`GRU_DM/`  
├── `Data/`  
├── `main_dem.py`  
├── `dem.py`*    
├── `train_dem.py`*    
├── `eval_dem.py`  
├── `utils.py`  
├── `dict_v0.py`*    
├── `dict.dict`*    

- `Data/`: Contains the datasets used for generating the BoW vectors, training the DM and evaluating the GRU-DM.  
- `main_dem.py`:  Pre-processes the test data. Main file for running the evaluation experiments of the GRU-DM.  
- `dem.py`:  Contains the GRU-DM model, and the two types of feedforward networks/classifiers (D3, D4, D5 and D6).  
- `train_dem.py`:  Pre-processes the train data and contains the training code of the GRU-DM.  
- `eval_dem.py`:  Contains the evaluation code of the GRU-DM.  
- `utils.py`:  Contains all the helper functions.
- `dict_v0.py`:  Contains code to generate the dictionary used for the GRU-DM.   
- `dict.dict`:  Contains the dictionary of the tokenizer used for GRU-DM.   

`Images/`  
- `Images/`: Stores the t-SNE visualizations of the evaluation of the DM.
  
`Models/`  
- `Models/`: Stores the DM after training.  

## Debiasing Method Experiments

`Debiasing/`  
├── `Data/`  
├── `dem.py`*    
├── `discriminators.py`  
├── `train_dialogue.py`  
├── `evaluate_dialogue.ipynb`  
├── `utils.py`  
├── `dict_v0.py`*    

- `Data/`: Contains the datasets used for training and evaluating the LLM.  
- `dem.py`: Contains the architecture of the GRU-DM for easy access.  
- `discriminators.py`: Contains the feedforward networks/classifiers (D1, D2).
- `train_dialogue.py`: Contains the training code of the LLM.
- `evaluate_dialogue.ipynb`: Contains the BLEU evaluation code of the LLM.    
- `utils.py`:  Contains all the helper functions.
- `dict_v0.py`:  Contains code to generate the dictionary used for the GRU-DM.  

`Models/` 
- `Models/`: Stores the fine-tuned LLM after training.

\* The code in these files integrates much code that was not made by me, but was retrieved from https://github.com/zgahhblhc/Debiased-Chat
