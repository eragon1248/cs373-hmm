# cs373-hmm

This code implements an HMM-based isolated word recognizer that is comprised of a signal processing feature extractor, a DNN model to classify  these features into phonemes, and an HMM recognizer. The recognizer is trained on a set of isolated words and then tested on a set of isolated words. The recognizer is evaluated using the word error rate (WER).

## Signal Processing
The code takes in a raw .wav file and loads it into a NumPy array. Then a number of operations are done to transform, filter, and discretize the signal into a set of feature frames. After computing the Fourier transform and binning with Mel-Filterbanks, the feature frames are reduced into an array of Mel-Frequency Cepstral Coefficients to be recognized by the DNN..

## Phoneme Classifier
The phoneme classifier is a DNN that takes in a set of MFCCs and outputs a probability distribution over the phonemes. The DNN is trained on a set of MFCCs and their corresponding phonemes. The DNN is implemented using PyTorch feedforward NN and is trained using the Adam optimizer.

## HMM Recognizer
Implemented an HMM-based isolated word recognizer. The recognizer uses the forward algorithm which computes the probability of a sequence of observations at any given timestep given a model. The outputs of this algorith is then used as inputs to the Viterbi algorithm which computes the most likely sequence of states given a sequence of observations. Finally a maximum likelihood update is performed to optimize the HMMs transition matrix. The recognizer is then evaluated using the word error rate (WER).