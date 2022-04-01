# LVH-THESIS
Classification of Left Ventricular Hyptertrophy using the multi-planar 3D nature of MRI images

The goal of my thesis project was to create a diagnosis aiding model specifically for the
detection and classification of left ventricular hypertrophy based on contrast enhanced
cardiac MRI images. The task was separated into and evaluated as two different classification
problems. One of them was a binary classification, in which I differentiated healthy,
or naturally enlarged (sport) hearts from the abnormally lesioned ones. While the other
one was a four-class classification, where the dataset was separated into Normal, Sport,
HCM (Hypertrophic cardiomyopathy) and other classes. I used both short- and long-axis
images for the solution, separating the latter one into three different views, and experimented
with a custom scar-segmentation approach to improve the overall performance of
the model.
The final, best results achieved on the test set were 91.89% accuracy and 91.84% F1-score
for the binary classification, and 78.38% accuracy and 77.74% F1-score for the four-class
classification problem.

Some parts of the dicom processing were heavily influenced by my project supervisors implementation (https://github.com/adamtiger/LHYP) and thus I've included the license from his repository accordingly.

The training and evaluation of the ensemble classifier was done using google colab. This part of the project is in the form of a Jupyter Notebook which makes it easier to follow and to run in a cloud environment.

As for the structure of the project:

------------
    ├── AutoEncoder    <- Model and scripts for the auto encoder
    │   ├── ae_dataset.py     <- Dataset class to use with pytorch dataloader
    │   ├── ae_model.py    <- Simple convolutional auto encoder model
    │   ├── ae_train.py    <- Training script for the auto encoder
    │
    ├── Classification    <- Jupyter notebooks for training and evaluating the ensemble classifier
    │   ├── ensemble_classification.ipynb     <- Notebook for training and evaluating the final ensemble classifier
    │   ├── ensemble_classification_tuning.ipynb    <- Classification with the addition of hyperparameter tuning using Ray Tune
    │
    ├── DicomPreprocess    <- Scripts and data-model for processing and serializing dicom data
    │   ├── dicom_reader_la.py       <- Helper class for classifying and converting long-axis dicom data
    │   ├── dicom_reader_sa.py       <- Helper class for converting short-axis dicom data
    │   ├── data_collector.py        <- Class for selecting, converting and serializing raw dicom data to useable patient data
    │   └── example_datacollect.py   <- Example script on how to use data_collector.py to process the raw dicom dataset
    │   └── DataModel 
    │      └── patient.py   <- The data model for the processed patient data
    │
    ├── TrainingPreprocess    <- Scripts for readying patient data to be used for training
    │   ├── filter_patients.py        <- Script for cleaning patient data leaving only the best frames for each axis
    │   ├── filtered_to_dataset.py    <- Script for structuring cleaned patient data into dataset folders
    │
    ├── Utils    <- Utility scripts and functions
    │   ├── stat_calc.py        <- Script to calculate the pth-quantile, mean and std of the dataset
    │   ├── utils.py            <- Helper functions for process logging
    ├── requirements.txt   <- The requirements file for reproducing the environment required to run the scripts
    │                       
    ├── thesis.pdf   <- Document explaining the steps of the project, results and the technical basics in detail. 
