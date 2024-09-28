# DPhil Dissertation 

Mac:
## Environment 
* Create a virtual environment with python 3.9.6: python 3.9 -m venv env
* Activate the environment in the terminal: source env/bin/activate 
* Install the requirements: pip3 install -r requirements.txt 

## Input 
* Input data file should be located in input/ 
* settings.yaml contains run settings 

## Code structure 
* The modules are located in src/ 
* read_data: read input data and construct time series 
* model: run requested model 
* data_analysis: runs diagnostic tests and create visualisations of time series; optional (see settings)
* utils: standalone functionality

## Usage 
* Run the following command from the folder root: python3 main.py 
* The folder tests/ contains some standalone analyses 

## Code Formatting and Standards
* A bash script doing the formatting is available, scripts/pre_commit.sh, put it in your pre-commit hook: 
    * Put '#!/bin/sh' (ENTER) 'source scripts/pre_commit.sh' in .git/hooks/pre-commit.sample
    * Rename pre-commit.sample to pre-commit: mv .git/hooks/pre-commit.sample .git/hooks/pre-commit
