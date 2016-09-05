# MusicGeneration

This is the MSc Final Project. The title of this project is Music Generation in Deep Learning. In this project, I try many different models, include existing ones and new ones to generate music and do comparison between them.

The source code of this project is on the Github: https://github.com/YuegeLI/MusicGeneration. All the source code can be ran directly. 

The 'fold data' includes all the data used in my project.

The fold 'model' includes the model RNN, LSTM, RNN-RBM, LSTM-RBM, AttLSTM-RBM and RF-Non Laplacian. Just run them directly. The generated music will be in the 'sample' fold.

The model GRU, MAF-Laplacian and MAF-Non Laplacian are in the fold 'MVA'. Use python notebook to run the code directly.

The code in the fold 'bleu' are used to calculate the bleu value. Just copy the generated music in this fold and run the code 'bleu.ipynb' with python notebook directly.
