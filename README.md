# progressiveNER
Neural Model for progressively learning new Named Entities in the text.
The model is implemented with TensorFlow in Python3. 

## Model
CRF over BLSTM

## Data Format
Existing models are trained on CoNLL 2003 NER dataset.
The data file is in the original CoNLL format

## Usage
For training (inital step):
```
python3 main.py --parameters_filepath=./parameters-train-step1.ini 
```
Hyperparameters are set in parameters-train-step1.ini and parameters-train-step2.ini for initial and subsequent step.
