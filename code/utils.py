from PIL import Image
import torch
from transformers import AutoImageProcessor, ResNetModel
from datasets import load_dataset, Image
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, roc_curve, auc
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import spacy
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_row(row):
    nlp = spacy.load('en_core_web_sm')
    text = row['text']
    doc = nlp(text)
    tokens = []
    for token in doc:
        pos = token.pos_
        lemma = token.lemma_
        tokens.append((token.text, lemma, pos))
    row['tokens'] = " ".join([t[0] for t in tokens])
    row['lemmas'] = " ".join([t[1] for t in tokens])
    row['upos'] = " ".join([t[2] for t in tokens])
    return row

def get_image_vectors(dataset, split, datafolder):
    """
    This function takes in a dataset and split (train, test or validation) and returns a list of image vectors.

    Args:
        dataset (dict): A dictionary containing the image paths and labels for each split.
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


# Define a function to get the vector representation of a text
def get_text_vector(text, tokenizer, model):
    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Move the inputs to the device
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
    # Tokenize input texts and create input tensors
    input_ids = []
    attention_masks = []
    labels = []
    for text, label in zip(df['text'], df['label']):
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

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Create dataset and dataloader
    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=32)

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 10
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Fine-tune model
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch_input_ids = batch[0].to(device)
            batch_attention_masks = batch[1].to(device)
            batch_labels = batch[2].to(device)
            model.zero_grad()
            loss, logits = model(batch_input_ids, token_type_ids=None, attention_mask=batch_attention_masks, labels=batch_labels, return_dict=False)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Finished epoch {epoch+1} with average training loss of {avg_train_loss}.")
    return tokenizer, model

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def predict_from_fine_tuned(df, tokenizer, model):
    input_ids = []
    attention_masks = []
    labels = []
    for text, label in zip(df['text'], df['label']):
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

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    val_sampler = SequentialSampler(dataset)
    val_dataloader = DataLoader(dataset, sampler=val_sampler, batch_size=32)

    # Evaluate model on validation data
    model.eval()
    total_val_accuracy = 0
    preds = []
    for batch in val_dataloader:
        batch_input_ids = batch[0].to(device)
        batch_attention_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)
        with torch.no_grad():
            outputs = model(batch_input_ids, token_type_ids=None, attention_mask=batch_attention_masks)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = batch_labels.to('cpu').numpy()
        preds.extend(np.argmax(logits, axis=1))
        total_val_accuracy += flat_accuracy(logits, label_ids)
    return preds


# Define function to classify text
def classify_text(text):
    # Tokenize input text
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt').to(device)

    # Make prediction with model
    model.eval()
    with torch.no_grad():
        output = model(input_ids)

    # Get predicted label
    predicted_label = torch.argmax(output[0], dim=1).item()

    # Return predicted label
    return predicted_label

# Get the vectors for each string in the list
def get_vectors(sentences, tokenizer, model):
    vector_list = []
    for text in sentences:
        text_vector = get_text_vector(text, tokenizer, model)
        vector_list.append(text_vector)
    return np.array(vector_list).reshape((len(vector_list), 768))

def get_f1(df):
    target = df['label']
    models = df.drop('label', axis=1)
    f1_scores = []
    for column in models.columns:
        f1 = f1_score(target, models[column], average='macro')
        f1_scores.append(f1)
    df_f1 = pd.DataFrame({'models': models.columns, 'F1-score': f1_scores}, index=None)
    return df_f1.sort_values(by='F1-score',  ascending=False)

def find_best_column_combination(df, column_names):
    best_score = 0.0
    best_columns = []
    
    for i in range(1, len(column_names) + 1):
        for cols in combinations(column_names, i):
            combined_col = df[list(cols)].apply(lambda row: any(row), axis=1)
            score = f1_score(df['label'], combined_col)
            if score > best_score:
                best_score = score
                best_columns = list(cols)
    
    return best_score, best_columns

def model_performance(df, models):
    results = []
    for model in models:
        y_true = df['label']
        y_pred = df[model]
        report = classification_report(y_true, y_pred, output_dict=True)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        AUROC = auc(fpr, tpr)
        results.append({'model': model, 'f1_score': report['macro avg']['f1-score'], 'precision': report['macro avg']['precision'], 'recall': report['macro avg']['recall'], 'accuracy': report['accuracy'], 'AUROC': AUROC})
        df_results = pd.DataFrame(results) 
        df_results = df_results.sort_values(by='AUROC', ascending=False)
    return df_results

def plot_disagreements(df, models, datafolder):
    """
    Function to plot images and model predictions for rows in a dataframe where the models disagree on the label.

    Parameters:
        df (pandas DataFrame): Input dataframe containing image data and model predictions
        models (list of strings): List of column names representing the different models used for prediction
        datafolder (string): Path to the folder containing the image files

    Returns:
        None

    Output:
        Displays a plot of images and model predictions for the top 10 rows where the models disagree on the label.
    """
    # Create a boolean mask for rows where the models disagree
    disagreements = (df[models].nunique(axis=1) > 1)
    # Get the indices of the top disagreements
    top_disagreements = disagreements.sort_values(ascending=False).head(10).index
    # Plot the images and model predictions for the top disagreements
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    for i, idx in enumerate(top_disagreements):
        ax = axes[i // 5][i % 5]
        img = plt.imread(datafolder+df.loc[idx, 'img'])
        ax.imshow(img)
        ax.axis('off')
        # ax.set_title(f"ID: {df.loc[idx, 'id']}")
        ax.text(0, -20, f"True label: {df.loc[idx, 'label']}", fontsize=14)
        for j, model in enumerate(models):
            pred = df.loc[idx, model]
            if pred == 1:
                ax.text(0, 25 + j * 20, f"{model[:4]}", color='red', fontsize=10)
            else:
                ax.text(0, 25 + j * 20, f"{model[:4]}", color='green', fontsize=10)
                
def stacked_ensemble(df_train, df_test, models, classifier):
    """
    Apply a stacked ensemble on a dataframe using a list of models and a simple classifier.
    Return the best combination of models and their performance metrics.
    LogisticRegression, RandomForestClassifier, or GradientBoostingClassifier.
    """
    # extract the output columns from the training and test dataframes
    X_train = df_train[models]
    X_test = df_test[models]
    
    # extract the target variable from the training dataframe
    y_train = df_train['label']
    
    # initialize variables to keep track of the best combination of models and its performance
    best_combo = None
    best_f1_score = 0.0
    classifier = classifier()
    # try different combinations of models
    for i in range(1, len(models) + 1):
        for combo in combinations(models, i):
            # extract the output columns corresponding to the current combination of models
            X_train_combo = X_train[list(combo)]
            X_test_combo = X_test[list(combo)]
            
            # train the simple classifier on the stacked features
            classifier.fit(X = X_train_combo, y = y_train)
            
            # make predictions on the test set
            y_pred = classifier.predict(X_test_combo)
            
            
            accuracy = accuracy_score(df_test['label'], y_pred)
            
            # update the best combination of models if necessary
            if accuracy > best_f1_score:
                # calculate the performance metrics
                f1 = f1_score(df_test['label'], y_pred, average='macro')
                precision = precision_score(df_test['label'], y_pred, average='macro')
                recall = recall_score(df_test['label'], y_pred, average='macro')
                best_f1_score = accuracy
                best_combo = combo
                best_performance = {'F1-score': f1, 'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}
                
    return best_combo, best_performance

                
def plot_most_significant_disagreements(df, models, datafolder):
    """
    Function to plot images and model predictions for rows in a dataframe where the models have the most significant disagreement on the label.

    Parameters:
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

    Parameters:
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