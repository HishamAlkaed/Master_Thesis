# Master's Thesis on the Role of Textual Modality in Hateful Memes Detection 

This repository is the culmination of my Master of Science degree in Text Mining at Vrije Universiteit Amsterdam, where I worked on the detection and classification of hateful memes. The dataset used in this project can be found online on Kaggle [here](https://www.kaggle.com/datasets/williamberrios/hateful-memes). Upon installation, save the data files in a directory called `data` one level higher than the readme file in this repository (the repository `Master_Thesis` and the data folder next to each other).  

## Requirments

Please refer to the `requirements.txt` file for all the needed packages to run the code in this repository. 

## Code

The `code` folder contains all the code needed for the experiments described in the thesis paper. All code is written using Python. Within the `code` folder, you will find the following files:



1. **preprocessing.ipynb:** This file contains all the preprocessing steps that were performed. Running this file sequentially should provide you with a version of the dataset that is compatible with the rest of the files in this repository. Make sure to save the output when finished.

2. **image_classifier.ipynb:** This file creates the image embeddings and then trains five machine learning models: one convolutional neural network (CNN) and four support vector machine (SVM) configurations.
3. **baseline.ipynb:** This file contains the implementation of unimodal bag-of-words (BoW) and character-n-grams models.

4. **advancedSVM.ipynb:** This file contains the implementation of the SVM model utilizing _stylometric & emotion-based_ and _transformer_ features.

5. **bert.ipynb:** This file contains the code used for creating three different BERT models: one using dehatebert directly, one utilizing word embeddings from the dehatebert model fed to an SVM, and one fine-tuning the BERT-base-cased model on the dataset.

6. **ensemble.ipynb:** This file contains the grid search performed to find the best combination of learners and meta-models, along with testing on the dataset's seen splits.

7. **neural_fusion.ipynb:** This file contains the implementation of late fusion using the MLP architecture explained in the thesis. Here, we fuse (the probabilities of) ResNet50 with BoW, character-n-grams, Fine-tuned BERT, and advanced features.

8. **error-analysis.ipynb:** This file contains the code used to perform the error analysis.

9. **ensemble_complete.ipynb:** This file provides an example of how to implement and run the ensemble model for the unseen split of the dataset. It covers the entire process from preprocessing to creating the models, training, and joining them. This could be used as an illustrative example of how to run a whole experiment from scratch. 

10. **plotting_features.ipynb:** This file contains some exploratory plots of the features created.

11. **utils.py:** This file provides helper functions.

## Conclusion

The various models implemented in this project demonstrate that it is possible to identify hateful memes using text and image analysis techniques. The unimodal fine-tuned BERT-base-cased model (utilizing only text) created in this project stands out as the most accurate, achieving an overall F1 score of 0.704 and 0.641 on the unseen dev and test split of the dataset, respectively. Followed by the Gradient Boosting Stacked Ensemble technique (utilizing both image and text), achieving 0.686 and 0.631 on the unseen dev and test split of the dataset, respectively. The insights behind this behaviour are explained in the thesis paper. The code in this repository, along with the dataset used and the thesis paper can be used as a starting point for future work in this field.
