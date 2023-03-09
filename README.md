# ChromPANACHE - Physics-based artificial neural network framework for adsorption and chromatography emulation
The python code ```TrainPANACHE.py``` trains the physics-based neural network model for simulating chromatographic columns without adsorption isotherms. The code follows the methodology provided in *Can a computer "learn" nonlinear chromatography?: Experimental validation of physics-based deep neural networks for  the simulation of chromatographic processes*. The example code provided herein was used to train the neural network model for simulating the column dynamics of more-retained enantiomer for the experimental case study considered in the paper. Note that this code can be utilized to train models for other case studies by simply changing the relevant coefficients. 


## FILE LIST
1. ```TrainPANACHE.py```: Trains physics-based neural networks for simulating the solute dynamics of chromatographic columns.
2. ```trainfcn.m```: Parser function loads ```train_data.mat``` and generates ```train_chrom.mat``` file required for running ```TrainPANACHE.py```. 
3. ```train_data.mat```: .mat data file contains the injection concentration and experimentally-measured elution profiles.

## SOFTWARE REQUIREMENTS AND INSTALLATION
### Dependencies 
The following dependencies are required for the proper execution of this program.
1. MATLAB version 2019b onwards [required]
2. Python 3 [required]
3. Tensorflow v1.15 (GPU) [required]

### Installation
1. Clone the full software package from the GitHub server into the preferred installation directory using: 
```
git clone https://github.com/ArvindRajendran/ChromPANACHE.git
```

## INSTRUCTIONS
1. Run trainfcn.m (with train_data.mat in the same directory) in MATLAB to generate train_chrom.mat.
2. Run TrainPANACHE.py (with train_chrom.mat in the same directory) in Python notebook 3.
3. Save the weights and biases of the trained model for subsequent use in model predictions. 

## CITATION
```
@article{Subraveti2023,
title = {Can a computer "learn" nonlinear chromatography?: Experimental validation of physics-based deep neural networks for  the simulation of chromatographic processes},
author = {Sai Gokul Subraveti and Zukui Li and Vinay Prasad and Arvind Rajendran},
journal = {ChemRxiv preprint},
year = {2023},
}
```

## AUTHORS 
###Maintainers of the repository 
- Sai Gokul Subraveti (subravet@ualberta.ca)

### Project Contributors 
- Prof. Dr. Arvind Rajendran (arvind.rajendran@ualberta.ca)
- Prof. Dr. Vinay Prasad (vprasad@ualberta.ca)
- Prof. Dr. Zukui Li (zukui@ualberta.ca)

## LICENSE 
Copyright (C) 2023 Arvind Rajendran

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.


