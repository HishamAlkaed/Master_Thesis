from PIL import Image
import torch
from transformers import AutoImageProcessor, ResNetModel
from datasets import load_dataset, Image
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd

def get_image_vectors(dataset, split):
    """
    This function takes in a dataset and split (train, test or validation) and returns a list of image vectors.

    Args:
        dataset (dict): A dictionary containing the image paths and labels for each split.
        split (str): The split to get image vectors for (train, test or validation).

    Returns:
        outputs_list (list): A list of image vectors for each image in the specified split.
    """

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
        results.append({'model': model, 'f1_score': report['macro avg']['f1-score'], 'precision': report['macro avg']['precision'], 'recall': report['macro avg']['recall'], 'accuracy': report['accuracy']})
        df_results = pd.DataFrame(results) 
        df_results = df_results.sort_values(by='f1_score', ascending=False)
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
            
            # calculate the performance metrics
            f1 = f1_score(df_test['label'], y_pred, average='macro')
            precision = precision_score(df_test['label'], y_pred, average='macro')
            recall = recall_score(df_test['label'], y_pred, average='macro')
            accuracy = accuracy_score(df_test['label'], y_pred)
            
            # update the best combination of models if necessary
            if accuracy > best_f1_score:
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