import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import joblib

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.LogSoftmax(dim=1)

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

# Load the model, encoder, and mappings
model_path = 'trained_model.pth'
encoder_path = 'trained_encoder.pkl'
metadata_path = 'metadata.pkl'
y_mapping_path = 'y_mapping.pkl'  # Path for y mapping

# Load metadata
metadata = joblib.load(metadata_path)
input_size = metadata['input_size']
output_size = metadata['output_size']

# Load encoder and y mapping
encoder = joblib.load(encoder_path)
y_mapping_dict = joblib.load(y_mapping_path)  # Load y mapping

# Initialize the model
model = NeuralNetwork(input_size, output_size)
model.load_state_dict(torch.load(model_path))
model.eval()

# Function to get BERT embeddings
def get_bert_embeddings(texts, tokenizer, bert_model):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(embedding)
    return np.vstack(embeddings)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Prediction function
def predict(user_input):
    input_df = pd.DataFrame(user_input)
    input_encoded = encoder.transform(input_df[['Operational Categorization Tier 1', 'Priority', 'Organization', 'Department']])
    input_summary_embeddings = get_bert_embeddings(input_df['Summary'].tolist(), tokenizer, bert_model)
    input_final = np.hstack((input_encoded, input_summary_embeddings))
    input_tensor = torch.tensor(input_final, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
        decoded_output = [y_mapping_dict[i] for i in predicted.numpy()]  # Decode the predicted class
    return decoded_output

# Example usage
user_input = {
    'Operational Categorization Tier 1': ['Failure'],
    'Summary': ['CHARIKA Issue- 6. One User'],
    'Priority': ['Medium'],
    'Organization': ['FLIGHT OPERATIONS'],
    'Department': ['FLIGHT CREW']
}

predicted_group = predict(user_input)
print(f'Predicted Assigned Group: {predicted_group}')
