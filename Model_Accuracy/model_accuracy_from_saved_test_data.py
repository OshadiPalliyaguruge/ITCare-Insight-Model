import torch
import numpy as np
import joblib
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score

# Load trained model metadata
metadata = joblib.load('your_model_metadata_path')
input_size = metadata['input_size']
output_size = metadata['output_size']

# Define the same model architecture
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, output_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return self.softmax(x)

# Load trained model
model = NeuralNetwork(input_size, output_size)
model.load_state_dict(torch.load('your_model_path'))
model.eval()

# Load trained encoder and label mappings
encoder = joblib.load('your_model_encoder_path')
y_mapping_dict = joblib.load('your_model_y_mapping_path')

# Function to get BERT embeddings (with added safety checks)
def get_bert_embeddings(texts, tokenizer, bert_model):
    embeddings = []
    for text in texts:
        # Ensure that text is a string and not None or unexpected type
        if isinstance(text, str):
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            outputs = bert_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            embeddings.append(embedding)
        else:
            # Handle non-string data (e.g., NaN, None, or invalid types)
            embeddings.append(np.zeros((1, 768)))  # Zero vector for non-string text (you may adjust this as needed)
    return np.vstack(embeddings)

# Load saved data
X_test = np.load('your_model_X_test.npy_path')
y_test = np.load('your_model_y_test.npy_path')
X_val = np.load('your_model_X_val.npy_path')
y_val = np.load('your_model_y_val.npy_path')

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Generate BERT embeddings for the test and validation sets
test_summary_embeddings = get_bert_embeddings(X_test[:, 1].astype(str), tokenizer, bert_model)
val_summary_embeddings = get_bert_embeddings(X_val[:, 1].astype(str), tokenizer, bert_model)

# Combine features for test and validation
X_test_final = np.hstack((X_test[:, :-1], test_summary_embeddings))  # Assuming categorical features are the first columns
X_val_final = np.hstack((X_val[:, :-1], val_summary_embeddings))

# Convert data to tensor
X_test_tensor = torch.tensor(X_test_final, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_final, dtype=torch.float32)

# Make predictions for test data
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, test_predicted = torch.max(test_outputs.data, 1)

# Make predictions for validation data
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    _, val_predicted = torch.max(val_outputs.data, 1)

# Convert predicted labels back to original categories
y_test_pred = [y_mapping_dict[i] for i in test_predicted.numpy()]
y_val_pred = [y_mapping_dict[i] for i in val_predicted.numpy()]

# Compute accuracy for test and validation data
test_accuracy = accuracy_score(y_test, y_test_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
