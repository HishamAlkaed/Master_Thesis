import pandas as pd
import spacy
import numpy as np
from datasets import Image
from itertools import combinations
from transformers import AutoImageProcessor, ResNetModel, AdamW, get_linear_schedule_with_warmup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, roc_curve, auc
from scipy.sparse import hstack
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_row(row):
    """
    Preprocesses the text in a single row of a DataFrame using spaCy

    Args:
        row (pandas.Series): A single row of DataFrame containing 'text' field

    Returns:
        pandas.Series : A new row added to DataFrame containing new columns for tokenised words, lemmas and
        Universal Part of Speech(word type) tags for the text in 'row'
    """
    # Load the spaCy pipeline
    nlp = spacy.load('en_core_web_sm')
    
    # Extract the text to be processed
    text = row['text']
    
    # Applying spaCy tokenizer
    doc = nlp(text)
   
    # Extracting processed values
    tokens = []
    for token in doc:
        pos = token.pos_
        lemma = token.lemma_
        tokens.append((token.text, lemma, pos))
        
    # Adding new columns to `row`
    row['tokens'] = " ".join([t[0] for t in tokens])
    row['lemmas'] = " ".join([t[1] for t in tokens])
    row['upos'] = " ".join([t[2] for t in tokens])
    
    return row

def get_image_vectors(dataset, split, datafolder):
    """
    This function takes in a dataset and split (train, test or validation) and returns a list of image vectors.

    Args:
        dataset (dict): A dictionary containing the image paths and labels for each split. It is advised to use TensorDataset for this.
        split (str): The split to get image vectors for (train, test or validation).

    Returns:
        outputs_list (list): A list of image vectors for each image in the specified split.
    """
    from PIL import Image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise use CPU
    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetModel.from_pretrained("microsoft/resnet-50").to(device)
    
    # Initialize list to store outputs
    outputs_list = []

    # Iterate through all images in the dataset
    for i in range(len(dataset[split]['img'])):

        # Open image using PIL
        image = Image.open(datafolder+dataset[split]['img'][i]['path'])
        
        # Check image format and convert if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Check image size and resize if necessary
        if image.size != (224, 224):
            image = image.resize((224, 224))
            
        # Preprocess image using the image processor
        inputs = image_processor(image, return_tensors="pt").to(device)

        # Pass image through the ResNet model
        with torch.no_grad():
            outputs = model(**inputs)

        # Append the last hidden state to the outputs list
        outputs_list.append(outputs.last_hidden_state.to(device))
    return outputs_list


def get_text_vector(text, tokenizer, model):
    """
    Generate the feature representation of the input text using the specified model

    Args:
        text (str): The input text
        tokenizer: Tokenizer object that can tokenize the input text
        model: Neural Network model that can generate feature vectors for the tokenized text

    Returns:
        numpy.ndarray: Feature representation (vector) of the input text
    """
    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Move the inputs to the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Get the output of the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the last hidden state of the BERT model
    last_hidden_state = outputs.last_hidden_state

    # Get the mean of the last hidden state across all tokens
    mean_last_hidden_state = torch.mean(last_hidden_state, dim=1)

    # Move the mean_last_hidden_state to the cpu and convert to a numpy array
    return mean_last_hidden_state.cpu().numpy()

def fine_tune(df, tokenizer, model):
    """
    Fine-tunes a pre-trained model by training it on a new dataset.

    Args:
        df (pandas DataFrame): A dataframe with columns 'text' and 'label', where 'text' is the input text and 
            'label' is the corresponding label.
        tokenizer (transformers.PreTrainedTokenizer): A pre-trained tokenizer.
        model (transformers.PreTrainedModel): A pre-trained model.

    Returns:
        A tuple of fine-tuned tokenizer (transformers.PreTrainedTokenizer) and fine-tuned model (transformers.PreTrainedModel).
    """

    # Tokenize input texts and create input tensors
    input_ids = []
    attention_masks = []
    labels = []
    for text, label in zip(df['text'], df['label']):
        # Tokenize the text using the provided tokenizer and get the input_ids and attention_mask tensors
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens = True,
                            max_length = 64,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(int(label))

    # Convert the lists of input_ids, attention_masks and labels to tensors and concatenate them vertically
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Create a TensorDataset from the input tensors and create a dataloader from the dataset
    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=32)

    # Set up the optimizer, scheduler, and the number of epochs
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 10
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Fine-tune the model by iterating over the epochs and the batches in each epoch
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch_input_ids = batch[0].to(device)
            batch_attention_masks = batch[1].to(device)
            batch_labels = batch[2].to(device)
            # Zero out the gradients, feed the batch to the model, calculate the loss, and backpropagate to get the gradients
            model.zero_grad()
            loss, logits = model(batch_input_ids, token_type_ids=None, attention_mask=batch_attention_masks, labels=batch_labels, return_dict=False)
            total_loss += loss.item()
            loss.backward()
            # Clip the gradients that exceed a threshold to deal with the "exploding gradient" problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update the model parameters and the learning rate
            optimizer.step()
            scheduler.step()
        # Print the average training loss for the epoch
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Finished epoch {epoch+1} with average training loss of {avg_train_loss}.")

    # Return the fine-tuned tokenizer and model
    return tokenizer, model

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    """
    Computes the accuracy of the predictions versus the true labels.

    Args:
        preds (numpy array): A 2D numpy array of shape `(n_samples, n_classes)` containing predicted scores 
            for each sample for each possible class.
        labels (numpy array): A 1D numpy array of shape `(n_samples,)` containing the correct labels for each sample.

    Returns:
        The accuracy of the predictions versus the true labels, as a float.    
    """
    
    # Compare the indices (i.e., labels) with the highest predicted scores for each sample with the true labels
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    # Count the number of correctly predicted samples and return the accuracy
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def predict_from_fine_tuned(df, tokenizer, model):
    """
    Uses a fine-tuned model to make predictions on a given dataframe.

    Args:
        df (pandas DataFrame): A dataframe with columns 'text' and 'label', where 'text' is the input text and 
            'label' is the corresponding label.
        tokenizer (transformers.PreTrainedTokenizer): A fine-tuned tokenizer.
        model (transformers.PreTrainedModel): A fine-tuned model.

    Returns:
        A list of prediction labels for the input dataframe.
    """

    # Tokenize input texts and create input tensors
    input_ids = []
    attention_masks = []
    labels = []
    for text, label in zip(df['text'], df['label']):
        # Tokenize the text using the provided tokenizer and get the input_ids and attention_mask tensors
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens = True,
                            max_length = 64,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(int(label))

    # Convert the lists of input_ids, attention_masks, and labels to tensors and concatenate them vertically
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Create a TensorDataset from the input tensors and create a dataloader from the dataset
    dataset = TensorDataset(input_ids, attention_masks, labels)
    val_sampler = SequentialSampler(dataset)
    val_dataloader = DataLoader(dataset, sampler=val_sampler, batch_size=32)

    # Evaluate the model on validation data
    model.eval()
    total_val_accuracy = 0
    preds = []
    for batch in val_dataloader:
        batch_input_ids = batch[0].to(device)
        batch_attention_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)
        with torch.no_grad():
            # Feed the batch to the model to get the predictions
            outputs = model(batch_input_ids, token_type_ids=None, attention_mask=batch_attention_masks)
        logits = outputs[0]
        # Convert the tensors to numpy ndarrays, then get the predicted labels and append them to "preds" list
        logits = logits.detach().cpu().numpy()
        label_ids = batch_labels.to('cpu').numpy()
        preds.extend(np.argmax(logits, axis=1))
        # Calculate the validation accuracy for the batch
        total_val_accuracy += flat_accuracy(logits, label_ids)

    # Return the list of predicted labels
    return preds



 # The probability at index `(i, 0)` is the probability that the `i`th text has `label=0`, and the probability at index `(i, 1)` is the probability that the `i`th text has `label=1`.
def predict_proba_from_fine_tuned(df, tokenizer, model):
    """
    Returns the predicted probabilities of the fine-tuned model for each text in the dataframe.

    Args:
        df (pandas DataFrame): A dataframe with columns 'text' and 'label', where 'text' is the input text and 
            'label' is the corresponding label.
        tokenizer (transformers.PreTrainedTokenizer): A pre-trained tokenizer.
        model (transformers.PreTrainedModel): A fine-tuned model.

    Returns:
        A two-dimensional numpy array with the probability of each text belonging to each class. 
        The probability at index (i, 0) is the probability that the ith text has label=0, and the probability at index 
        (i, 1) is the probability that the ith text has label=1.
    """

    input_ids = []
    attention_masks = []
    labels = []
    for text, label in zip(df['text'], df['label']):
        # Tokenize the text using the provided tokenizer and get the input_ids and attention_mask tensors
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens = True,
                            max_length = 64,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(int(label))

    # Convert the lists of input_ids, attention_masks and labels to tensors and concatenate them vertically
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Create a TensorDataset from the input tensors and create a dataloader from the dataset
    dataset = TensorDataset(input_ids, attention_masks, labels)
    val_sampler = SequentialSampler(dataset)
    val_dataloader = DataLoader(dataset, sampler=val_sampler, batch_size=32)

    # Evaluate the fine-tuned model on the validation data to get the predicted probabilities
    model.eval()
    total_val_accuracy = 0
    preds = []
    probabilities = []
    for batch in val_dataloader:
        batch_input_ids = batch[0].to(device)
        batch_attention_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)
        # Get the predictions and the probabilities for the batch using the fine-tuned model
        with torch.no_grad():
            outputs = model(batch_input_ids, token_type_ids=None, attention_mask=batch_attention_masks)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = batch_labels.to('cpu').numpy()
        batch_probs = softmax(logits, axis=-1)
        # Extend the lists of probabilities and predictions for all the batches
        preds.extend(np.argmax(logits, axis=1))
        probabilities.extend(batch_probs)

    # Convert the list of probabilities to a numpy array
    probabilities = np.asarray(probabilities)
    return probabilities

def classify_text(text):
    """
    Classifies the input text using the fine-tuned model.

    Args:
        text (str): Input text to classify.

    Returns:
        The predicted label for the input text.
    """
    # Encode the text using the provided tokenizer and convert to tensor
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt').to(device)

    # Make a prediction on the input text using the fine-tuned model
    model.eval()
    with torch.no_grad():
        output = model(input_ids)

    # Get the predicted label by taking the argmax of the output tensor
    predicted_label = torch.argmax(output[0], dim=1).item()

    # Return the predicted label
    return predicted_label

def get_vectors(sentences, tokenizer, model):
    """
    Returns the embeddings of each string in the list of sentences.

    Args:
        sentences (list of str): A list of sentences or strings.
        tokenizer (transformers.PreTrainedTokenizer): A pre-trained tokenizer.
        model (transformers.PreTrainedModel): A pre-trained model.

    Returns:
        A two-dimensional numpy array with the embeddings of each sentence or string. The array has shape (m, 768),
        where m is the number of sentences or strings in the list, and 768 is the dimensionality of the embeddings.
    """
    vector_list = []
    for text in sentences:
        # Get the embedding vector for the current text using the get_text_vector function
        text_vector = get_text_vector(text, tokenizer, model)
        vector_list.append(text_vector)
    # Convert the list of embeddings to a numpy array and reshape it to (m, 768)
    return np.array(vector_list).reshape((len(vector_list), 768))

def get_f1(df):
    """
    Computes the macro F1 score of each model in a dataframe of predicted labels and returns a sorted dataframe
    with the F1 scores in descending order.

    Args:
        df (pandas DataFrame): A dataframe where each column corresponds to a different model and each row corresponds
        to a different text. The values in the dataframe are the predicted labels of the corresponding model for the
        corresponding text.

    Returns:
        A dataframe with two columns: 'models' and 'F1-score'. The 'models' column contains the names of the models in the
        original dataframe, and the 'F1-score' column contains the macro F1 score of each model in descending order.
    """
    target = df['label']
    # Extract the predicted labels for each model in the dataframe
    models = df.drop('label', axis=1)
    f1_scores = []
    # Compute the macro F1 score of each model and add it to a list
    for column in models.columns:
        f1 = f1_score(target, models[column], average='macro')
        f1_scores.append(f1)
    # Create a new dataframe with the models and their corresponding F1 scores, and sort it by descending F1 score
    df_f1 = pd.DataFrame({'models': models.columns, 'F1-score': f1_scores}, index=None)
    return df_f1.sort_values(by='F1-score',  ascending=False)

def find_best_column_combination(df, column_names):
    """
    Finds the best combination of columns in a pandas dataframe to maximize the macro F1 score of the resulting
    binary classification model. The fusion here is logical OR. this is just used as a trail.

    Args:
        df (pandas DataFrame): A dataframe where each column corresponds to a different feature and each row corresponds
        to a different text. The values in the dataframe are the features for the corresponding text.
        column_names (list of str): A list of the column names to consider for the combination.

    Returns:
        A tuple containing the best F1 score and the list of column names that produced it.
    """
    best_score = 0.0
    best_columns = []
    
    # Iterate over all possible combinations of columns
    for i in range(1, len(column_names) + 1):
        for cols in combinations(column_names, i):
            # Combine the selected columns using a logical OR operator to create a new binary column
            combined_col = df[list(cols)].apply(lambda row: any(row), axis=1)
            # Compute the F1 score of the resulting binary classifier and update the best score if necessary
            score = f1_score(df['label'], combined_col)
            if score > best_score:
                best_score = score
                best_columns = list(cols)
    
    return best_score, best_columns

def model_performance(df, models):
    """
    Evaluates the performance of each model in a pandas dataframe using several metrics and returns a sorted dataframe
    with the results, sorted by AUROC.

    Args:
        df (pandas DataFrame): A dataframe where each column corresponds to a different model and each row corresponds
        to a different text. The values in the dataframe are the predicted labels of the corresponding model for the
        corresponding text.
        models (list of str): A list of model names (column names in the dataframe) to evaluate.

    Returns:
        A dataframe with the performance results for each model. The dataframe has one row per model and the following
        columns: 'model', 'f1_score', 'precision', 'recall', 'accuracy', 'AUROC'. The rows are sorted by descending AUROC.
    """
    results = []
    # Iterate over each model in the list of model names
    for model in models:
        y_true = df['label']
        y_pred = df[model]
        # Compute the classification report and AUROC of the current model
        report = classification_report(y_true, y_pred, output_dict=True)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        AUROC = auc(fpr, tpr)
        # Add the performance results to a list
        results.append({'model': model, 'f1_score': report['macro avg']['f1-score'],
                        'precision': report['macro avg']['precision'], 'recall': report['macro avg']['recall'],
                        'accuracy': report['accuracy'], 'AUROC': AUROC})
        # Create a dataframe with the performance results and sort by descending AUROC
        df_results = pd.DataFrame(results) 
        df_results = df_results.sort_values(by='AUROC', ascending=False)
    return df_results

def plot_disagreements(df, models, datafolder):
    """
    Plots the images and model predictions for rows in a dataframe where the models disagree on the label.

    Args:
        df (pandas DataFrame): A dataframe containing image data and model predictions.
        models (list of str): A list of column names representing the different models used for prediction.
        datafolder (str): The path to the folder containing the image files.

    Returns:
        None. 

    Output:
        A plot of images and model predictions for the top 10 rows where the models disagree on the label.
    """

    # Create a boolean mask for rows where the models disagree
    disagreements = (df[models].nunique(axis=1) > 1)

    # Get the indices of the top disagreements
    top_disagreements = disagreements.sort_values(ascending=False).head(10).index # We take the top 10 rows

    # Plot the images and model predictions for the top disagreements
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))

    for i, idx in enumerate(top_disagreements):
        ax = axes[i // 5][i % 5]
        img = plt.imread(os.path.join(datafolder, df.loc[idx, 'img'])) # Load image
        ax.imshow(img)
        ax.axis('off')
        ax.text(0, -20, f"True label: {df.loc[idx, 'label']}", fontsize=14)
        for j, model in enumerate(models):
            pred = df.loc[idx, model]
            # Color the model's abbreviation in red if the prediction is positive, green if negative
            if pred == 1:
                ax.text(0, 25 + j * 20, f"{model[:4]}", color='red', fontsize=10)
            else:
                ax.text(0, 25 + j * 20, f"{model[:4]}", color='green', fontsize=10)
                
def stacked_ensemble(df_train, df_test, models, classifier):
    """
    Trains a stacked ensemble of models using a simple classifier and returns the best combination of models and their
    performance metrics.

    Args:
        - df_train (pandas DataFrame): A dataframe where each column corresponds to a different model and each row
        corresponds to a different example in the training set. The values in the dataframe are the predicted labels of
        the corresponding model for the corresponding example.
        - df_test (pandas DataFrame): A dataframe with the same structure as df_train representing the test set.
        models (list of str): A list of the column names in df_train and df_test that correspond to the models to be used
        in the stacked ensemble.
        - classifier (class constructor): A constructor for the classifier to be used in the stacked ensemble. Must take
        no arguments.

    Returns:
        A tuple containing the best combination of models in the ensemble (a list of column names) and their performance
        metrics (a dictionary with keys 'F1-score', 'Precision', 'Recall', 'Accuracy', and 'AUROC').
    """
    # Extract the output columns from the training and test dataframes
    X_train = df_train[models]
    X_test = df_test[models]
    
    # Extract the target variable from the training dataframe
    y_train = df_train['label']
    
    # Initialize variables to keep track of the best combination of models and its performance
    best_combo = None
    best_accuracy = 0.0
    classifier = classifier() # Create an instance of the given classifier
    # Try different combinations of models
    for i in range(1, len(models) + 1):
        for combo in combinations(models, i):
            # Extract the output columns corresponding to the current combination of models
            X_train_combo = X_train[list(combo)]
            X_test_combo = X_test[list(combo)]
            
            # Train the simple classifier on the stacked features
            classifier.fit(X_train_combo, y_train)
            
            # Make predictions on the test set
            y_pred = classifier.predict(X_test_combo)
            
            # Compute the accuracy score, as it is the metric to maximize
            accuracy = accuracy_score(df_test['label'], y_pred)
            
            # Update the best combination of models if necessary
            if accuracy > best_accuracy:
                # Calculate the performance metrics
                f1 = f1_score(df_test['label'], y_pred, average='macro')
                precision = precision_score(df_test['label'], y_pred, average='macro')
                recall = recall_score(df_test['label'], y_pred, average='macro')
                fpr, tpr, thresholds = roc_curve(df_test['label'], y_pred)
                AUROC = auc(fpr, tpr)
                best_accuracy = accuracy
                best_combo = combo
                best_performance = {'F1-score': f1, 'Precision': precision, 'Recall': recall, 'Accuracy': accuracy, 'AUROC':AUROC}
                
    return best_combo, best_performance
                
def plot_most_significant_disagreements(df, models, datafolder):
    """
    Function to plot images and model predictions for rows in a dataframe where the models have the most significant disagreement on the label.

    Args:
        df (pandas DataFrame): Input dataframe containing image data and model predictions
        models (list of strings): List of column names representing the different models used for prediction
        datafolder (string): Path to the folder containing the image files

    Returns:
        None

    Output:
        Displays a plot of images and model predictions for the top 10 rows where the models have the most significant disagreement on the label.
    """
    # Create a boolean mask for rows where the models have equal number of true and false predictions
    equal_disagreements = ((df[models] == 1).sum(axis=1) == (len(models) / 2))
    # Create a boolean mask for rows where the models disagree
    disagreements = (df[models].nunique(axis=1) > 2)
    # Combine the two masks to get the rows where there is the most significant disagreement
    # significant_disagreements = equal_disagreements & disagreements
    significant_disagreements = disagreements 
    # Get the indices of the top disagreements
    top_disagreements = significant_disagreements.sort_values(ascending=False).head(10).index
    # Plot the images and model predictions for the top disagreements
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    for i, idx in enumerate(top_disagreements):
        ax = axes[i // 5][i % 5]
        img = plt.imread(datafolder+df.loc[idx, 'img'])
        ax.imshow(img)
        ax.axis('off')
        true_label = df.loc[idx, 'label']
        ax.text(0, -20, f"True label: {true_label}", fontsize=14)
        for j, model in enumerate(models):
            pred = df.loc[idx, model]
            if pred == 1:
                ax.text(0, 25 + j * 20, f"{model[:4]}", color='red', fontsize=10)
            else:
                ax.text(0, 25 + j * 20, f"{model[:4]}", color='green', fontsize=10)
                
def plot_disagreements_by_true(df, models, datafolder, num_true):
    """
    Function to plot images and model predictions for rows in a dataframe where the models disagree and have a specific number of true predictions.

    Args:
        df (pandas DataFrame): Input dataframe containing image data and model predictions
        models (list of strings): List of column names representing the different models used for prediction
        datafolder (string): Path to the folder containing the image files
        num_true (int): The number of true predictions that the row should have

    Returns:
        None

    Output:
        Displays a plot of images and model predictions for the rows where the models disagree and have the specified number of true predictions.
    """
    # Create a boolean mask for rows where there is a specific number of true predictions
    true_mask = ((df[models] == 1).sum(axis=1) == num_true)
    # Create a boolean mask for rows where the models disagree
    disagreement_mask = (df[models].nunique(axis=1) > 1)
    # Combine the two masks to get the rows where there is disagreement and the specified number of true predictions
    selected_rows = true_mask & disagreement_mask
    # Get the indices of the selected rows
    selected_indices = selected_rows[selected_rows].index
    # Plot the images and model predictions for the selected rows
    num_plots = min(10, len(selected_indices))
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    for i, idx in enumerate(selected_indices[:num_plots]):
        ax = axes[i // 5][i % 5]
        img = plt.imread(datafolder+df.loc[idx, 'img'])
        ax.imshow(img)
        ax.axis('off')
        true_label = df.loc[idx, 'label']
        ax.text(0, -20, f"True label: {true_label}", fontsize=14)
        for j, model in enumerate(models):
            pred = df.loc[idx, model]
            if pred == 1:
                ax.text(0, 25 + j * 20, f"{model[:4]}", color='red', fontsize=10)
            else:
                ax.text(0, 25 + j * 20, f"{model[:4]}", color='green', fontsize=10)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # Define the layers of the neural network
        self.flattening = nn.Flatten() # Flatten the input tensor before passing it through linear layers
        self.fc1 = nn.Linear(4, 8) # First linear layer with input size 4 and output size 8
        self.fc2 = nn.Linear(8, 4) # Second linear layer with input size 8 and output size 4
        self.dropout = nn.Dropout(p=0.5) # Dropout layer with probability of 0.5
        self.fc3 = nn.Linear(4, 2) # Third linear layer with input size 4 and output size 2

    def forward(self, x):
        # Define the forward pass for the neural network
        x = self.flattening(x)
        x = F.relu(self.fc1(x)) # Apply the ReLU activation function to the output of the first linear layer
        x = F.relu(self.fc2(x)) # Apply the ReLU activation function to the output of the second linear layer
        x = self.dropout(x) # Apply the dropout layer to the output of the second linear layer
        x = F.softmax(self.fc3(x), dim=1) # Apply the softmax function to the output of the third linear layer
        return x
    
def fuse_proba(arr1, arr2):
    """
    Concatenates two numpy arrays horizontally.

    Args:
        arr1 (numpy array): The first numpy array to concatenate.
        arr2 (numpy array): The second numpy array to concatenate.

    Returns:
        A numpy array obtained by horizontally stacking arr1 and arr2.
    """
    # Check that the input arrays have the same size
    assert arr1.size == arr2.size
    # Concatenate the arrays horizontally using np.hstack
    return np.hstack([arr1, arr2])

def train(X, Y, batch_size = 64, num_epochs = 1000):
    """
    Trains a neural network classifier using binary cross-entropy loss and the Adam optimizer.

    Args:
        X (numpy array): The input data of shape (num_samples, num_features).
        Y (numpy array): The binary labels of shape (num_samples,).
        batch_size (int): The number of samples to use for each batch during training.
        num_epochs (int): The number of training epochs.

    Returns:
        The trained neural network model.
    """
    # Define the neural network model, loss function and optimizer
    model = MLP() # MLP is defined in the previous example
    criterion = nn.BCELoss() # Binary cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters()) # Adam optimizer for gradient descent

    # Create a PyTorch data loader for efficient data loading
    train_dataset = TensorDataset(torch.Tensor(X), torch.Tensor(Y))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Training loop:
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Zero the gradients before forward and backward pass
            optimizer.zero_grad()
            outputs = model(inputs) # Predict the class probabilities for the input samples
            labels = labels.view(-1, 1)  # Reshape labels to match output shape
            # Calculate the loss using binary cross-entropy with the positive class probability and the binary label
            loss = criterion(torch.unsqueeze(outputs[:, 1], dim=1), labels)
            # Compute the gradients and update the weights using backpropagation
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return model

def evaluate(model, X, Y):
    """
    Evaluates a neural network classifier by computing the binary predictions and probabilities on a test set.

    Args:
        model (PyTorch model): The trained neural network model.
        X (numpy array): The input test data of shape (num_samples, num_features).
        Y (numpy array): The binary labels of the test data of shape (num_samples,).

    Returns:
        The binary predictions of the neural network classifier on the test set.
    """
    test_dataset = TensorDataset(torch.Tensor(X), torch.Tensor(Y))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # Set the model to evaluation mode
    model.eval()
    # Disable gradient calculation
    with torch.no_grad():
        for inputs, labels in test_loader:
            test_outputs = model(inputs)  # Get the model's predictions
            _, predicted = torch.max(test_outputs.data, 1)  # Get the predicted class by choosing the class with highest probability
            predicted_probabilities = test_outputs[:, 1]  # Get the probability for the positive class

    # Apply a threshold to the predicted probabilities to obtain binary predictions
    threshold = 0.5
    binary_predictions = (predicted_probabilities > threshold).float()

    return binary_predictions

def performance(preds, labels):
    """
    Computes and returns the performance metrics of a binary classifier given its binary predictions and labels.

    Args:
        preds (numpy array): The binary predictions of the binary classifier with shape (num_samples,).
        labels (numpy array): The binary labels of the test data with shape (num_samples,).

    Returns:
        A pandas DataFrame containing the performance metrics of the binary classifier: 
        f1_score, precision, recall, accuracy, and AUROC.
    """
    results = []
    # Compute the classification report containing the precision, recall, and f1-score for each class and their averages
    report = classification_report(labels, preds, output_dict=True)
    # Compute the false positive rate, true positive rate, and thresholds using the receiver operating characteristic (ROC) curve
    fpr, tpr, thresholds = roc_curve(labels, preds)
    # Compute the area under ROC curve (AUROC)
    AUROC = auc(fpr, tpr)
    # Add the performance metrics to a dictionary
    results = {'f1_score': report['macro avg']['f1-score'], 
               'precision': report['macro avg']['precision'], 
               'recall': report['macro avg']['recall'], 
               'accuracy': report['accuracy'], 
               'AUROC': AUROC
              }
    # Convert the results dictionary to a pandas DataFrame
    df_results = pd.DataFrame(results, index=[0]) 
    # Sort the results by the AUROC metric in descending order
    df_results = df_results.sort_values(by='AUROC', ascending=False)
    return df_results

def late_fuse_MLP(X_train, Y_train, X_test, Y_test):
    """
    Trains a neural network classifier on the training data and evaluates it on the test data.

    Args:
        X_train (numpy array): The input training data of shape (num_train_samples, num_features).
        Y_train (numpy array): The binary labels of the training data of shape (num_train_samples,).
        X_test (numpy array): The input test data data of shape (num_test_samples, num_features).
        Y_test (numpy array): The binary labels of the test data of shape (num_test_samples,).

    Returns:
        A pandas DataFrame containing the performance metrics of the trained neural network classifier on the test set: 
        f1_score, precision, recall, accuracy, and AUROC.
    """
    # Train a neural network on the training data
    model = train(X_train, Y_train)
    # Use the trained neural network to make binary predictions on the test data
    pred_test = evaluate(model, X_test, Y_test)
    # Compute the performance metrics of the neural network classifier on the test set
    test_results_df = performance(pred_test, Y_test)
    return test_results_df

def late_fuse_MLPX(X_train, Y_train, X_dev, Y_dev, X_test, Y_test):
    """
    Trains a neural network classifier on the training data, evaluates it on the development and test sets, 
    and returns the performance metrics on both sets.

    Args:
        X_train (numpy array): The input training data of shape (num_train_samples, num_features).
        Y_train (numpy array): The binary labels of the training data of shape (num_train_samples,).
        X_dev (numpy array): The input development data of shape (num_dev_samples, num_features).
        Y_dev (numpy array): The binary labels of the development data of shape (num_dev_samples,).
        X_test (numpy array): The input test data of shape (num_test_samples, num_features).
        Y_test (numpy array): The binary labels of the test data of shape (num_test_samples,).

    Returns:
        Two pandas DataFrames containing the performance metrics of the trained neural network classifier: 
        f1_score, precision, recall, accuracy, and AUROC on the development and test sets, respectively.
    """
    # Train a neural network on the training data
    model = train(X_train, Y_train)
    # Use the trained neural network to make binary predictions on the development and test data
    pred_dev = evaluate(model, X_dev, Y_dev)
    pred_test = evaluate(model, X_test, Y_test)
    # Compute the performance metrics of the neural network classifier on the development and test sets
    dev_results_df = performance(pred_dev, Y_dev)
    test_results_df = performance(pred_test, Y_test)
    # Print the performance metrics on both sets
    print(f'Development {dev_results_df} \n Test {test_results_df}')
    return dev_results_df, test_results_df

def run_multiple_times(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, n=10):
    """
    Trains and evaluates a neural network classifier multiple times on the development and test sets,
    and returns the average performance metrics on both sets.

    Args:
        X_train (numpy array): The input training data of shape (num_train_samples, num_features).
        Y_train (numpy array): The binary labels of the training data of shape (num_train_samples,).
        X_dev (numpy array): The input development data of shape (num_dev_samples, num_features).
        Y_dev (numpy array): The binary labels of the development data of shape (num_dev_samples,).
        X_test (numpy array): The input test data of shape (num_test_samples, num_features).
        Y_test (numpy array): The binary labels of the test data of shape (num_test_samples,).
        n (int): The number of times to train and evaluate the classifier.

    Returns:
        Two pandas DataFrames containing the average performance metrics of the trained neural network classifier: 
        f1_score, precision, recall, accuracy, and AUROC on the development and test sets, respectively.
    """
    dev_results_list = []
    test_results_list = []
    # Train and evaluate the neural network classifier n times
    for i in range(n):
        dev_results, test_results = late_fuse_MLPX(X_train, Y_train, X_dev, Y_dev, X_test, Y_test)
        dev_results_list.append(dev_results)
        test_results_list.append(test_results)
    # Compute the average performance metrics on the development and test sets over n iterations
    dev_results_avg = pd.concat(dev_results_list).groupby(level=0).mean()
    test_results_avg = pd.concat(test_results_list).groupby(level=0).mean()
    # Print the average performance metrics on both sets
    print(f'Average Development: \n{dev_results_avg} \nAverage Test: \n{test_results_avg}')
    return dev_results_avg, test_results_avg
    
def plot_confusion_matrix(df, label_col, pred_col, path_to_save=None):
    """
    Function to plot a confusion matrix given a DataFrame with true labels and model predictions.

    Args:
        df (pandas DataFrame): Input dataframe containing true labels and model predictions
        label_col (string): Name of column containing the true labels
        pred_col (string): Name of column containing the model predictions
        path_to_save (string or None): Optional path to save the plot as an image. Default is None (no save).

    Returns:
        None

    Output:
        Displays a plot of the confusion matrix. If path_to_save is provided, saves the plot as an image.
    """
    # Calculate the confusion matrix values
    tn = ((df[label_col] == 0) & (df[pred_col] == 0)).sum()
    fp = ((df[label_col] == 0) & (df[pred_col] == 1)).sum()
    fn = ((df[label_col] == 1) & (df[pred_col] == 0)).sum()
    tp = ((df[label_col] == 1) & (df[pred_col] == 1)).sum()

    # Create the confusion matrix as a numpy array
    cm = np.array([[tn, fp], [fn, tp]])

    # Set up the plot
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap='Blues')

    # Add the values to the plot
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=18)

    # Add labels and legend to the plot
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax)

    # Save or show the plot
    if path_to_save:
        plt.savefig(path_to_save, bbox_inches='tight', pad_inches=0.1, dpi = 300)
    else:
        plt.show()
    
def plot_image(image_path):
    """
    Function to plot a single image given the image path.

    Args:
        image_path (string): Path to the image file.

    Returns:
        None

    Output:
        Displays a plot of the image.
    """
    # Load the image
    img = plt.imread(image_path)
    # Plot the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
def plot_images(image_paths, file_path = ''):
    """
    Function to plot multiple images given a list of image paths.

    Args:
        image_paths (list of strings): List of paths to image files.

    Returns:
        None

    Output:
        Displays a plot of the images.
    """
    # Set up the subplot grid
    n_cols = min(3, len(image_paths))
    n_rows = (len(image_paths) - 1) // n_cols + 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 8))

    # Check if axes is one-dimensional and reshape if necessary
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot each image
    for i, path in enumerate(image_paths):
        ax = axes[i // n_cols, i % n_cols]
        img = plt.imread(file_path+path)
        ax.imshow(img)
        ax.axis('off')
        # ax.set_title(path)

    # Show the plot
    plt.show()
    return fig

def plot_diff_images(image_paths, df, file_path=''):
    """
    Function to plot multiple images given a list of image paths and annotate with model predictions and true label.

    Args:
        image_paths (list of strings): List of paths to image files.
        df (pandas DataFrame): DataFrame containing columns "bert_base_cased_finetuned", "GBensemble", "bert+resnet",
        "img" and "label".
        file_path (string): Path to the directory where the images are located.

    Returns:
        None

    Output:
        Displays a plot of the images with legend showing the predictions for each model and the true label.
    """
    # Set up the subplot grid
    n_cols = 2  # always display 2 images per row
    n_rows = (len(image_paths) - 1) // n_cols + 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 8))

    # Check if axes is one-dimensional and reshape if necessary
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot each image
    for i, path in enumerate(image_paths):
        ax = axes[i // n_cols, i % n_cols]
        img = plt.imread(file_path + path)
        ax.imshow(img)
        ax.axis('off')
        # annotate with model predictions and true label
        label = df.loc[df['img'] == path, 'label'].values[0]
        ax.set_title('True Label: ' + str(label), fontsize=14)
        prediction_str = 'Predictions:\n'
        for model in ["bert_base_cased_finetuned", "GBensemble", "bert+resnet"]:
            prediction_str += f"{model}: {df.loc[df['img'] == path, model].values[0]}\n"
        prediction_str = prediction_str.rstrip('\n')  # remove trailing newline character
        ax.text(0.5, -0.1, prediction_str, verticalalignment='bottom',
        horizontalalignment='center', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.9),
        fontsize=14)
    # Remove empty axes
    for i in range(len(image_paths), n_rows * n_cols):
        axes.flatten()[-i - 1].remove()


    plt.tight_layout()

    # Show the plot
    plt.show()
    return fig
    
def error_analysis(data, model):
    """
    Perform error analysis on a dataset of binary classification by finding the tps, tns, fps, and fns.

    Args:
        data (pandas DataFrame): The dataset containing the model predictions and gold labels.
        model (str): The name of the model predictions column in the DataFrame.

    Returns:
        Four pandas DataFrames containing the false positives, false negatives, true positives, and true negatives.
    """
    # Create a dictionary to store the results
    results = {}
    
    # Get the false positives: predicted positive but actually negative
    false_positives = data[(data[model] == 1) & (data['label'] == 0)]
    
    # Get the false negatives: predicted negative but actually positive
    false_negatives = data[(data[model] == 0) & (data['label'] == 1)]
    
    # Get the true negatives: predicted negative and actually negative
    true_negatives = data[(data[model] == 0) & (data['label'] == 0)]
    
    # Get the true positives: predicted positive and actually positive
    true_positives = data[(data[model] == 1) & (data['label'] == 1)]
    
    return false_positives, false_negatives, true_positives, true_negatives