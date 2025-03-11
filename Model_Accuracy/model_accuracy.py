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

# Function to get BERT embeddings
def get_bert_embeddings(texts, tokenizer, bert_model):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(embedding)
    return np.vstack(embeddings)

# Load the dataset
data = pd.read_csv('your_dataset_path')

# Select a portion for testing (e.g., last 20% of data)
test_size = int(len(data) * 0.2)
data_test = data.iloc[-test_size:]  

# Prepare input features
X_test = data_test[['Operational Categorization Tier 1', 'Summary', 'Priority', 'Organization', 'Department']]
y_test = data_test['Assigned Group']

# One-hot encode categorical features
X_encoded = encoder.transform(X_test[['Operational Categorization Tier 1', 'Priority', 'Organization', 'Department']])

# Generate BERT embeddings for summary column
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
summary_embeddings = get_bert_embeddings(X_test['Summary'].tolist(), tokenizer, bert_model)

# Combine features
X_final = np.hstack((X_encoded, summary_embeddings))

# Convert test data to tensor
X_test_tensor = torch.tensor(X_final, dtype=torch.float32)

# Make predictions using the trained model
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)

# Convert predicted labels back to original categories
y_pred = [y_mapping_dict[i] for i in predicted.numpy()]
y_test = y_test.tolist()

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
