import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split


# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)

# Function to get bert embeddng
def get_bert_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    return np.array(embeddings).squeeze()

# Function for making predictions based on new input
def predict(input_data):
    input_df = pd.DataFrame(input_data)
    
    # Handle unknown categories
    try:
        input_encoded = encoder.transform(input_df[['Operational Categorization Tier 1', 'Priority', 'Organization', 'Department']])
    except ValueError as e:
        print(f'Error in encoding input data: {e}')
        return None

    input_summary_embeddings = get_bert_embeddings(input_df['Summary'].tolist())
    input_final = np.hstack((input_encoded, input_summary_embeddings))
    input_tensor = torch.tensor(input_final, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
    return encoder.inverse_transform(predicted.numpy().reshape(-1, 1))

# Load the dataset
data = pd.read_csv('incident_report_preprocessed_final_98000_cleaned.csv')

# Assume the dataset contains 'Operational Categorization Tier 1' for X and 'Assigned Group' for Y
X = data[['Operational Categorization Tier 1', 'Summary', 'Priority', 'Organization', 'Department']]
y = data['Assigned Group']

# Prepare BERT embeddings for the text features
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Generate embeddings for the Summary column
summary_embeddings = get_bert_embeddings(X['Summary'].tolist())

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)  # Updated parameter
X_encoded = encoder.fit_transform(X[['Operational Categorization Tier 1', 'Priority', 'Organization', 'Department']])

# Combine the one-hot encoded features with BERT embeddings
X_final = np.hstack((X_encoded, summary_embeddings))

# Map y to numeric values
y_encoded = pd.factorize(y)[0]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y_encoded, test_size=0.2, random_state=42)

# Initialize the model, define loss function and optimizer
input_size = X_final.shape[1]
output_size = len(np.unique(y_encoded))  # Number of unique labels
model = NeuralNetwork(input_size, output_size)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Testing the model
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Accuracy of the model: {accuracy * 100:.2f}%')


# Example usage
input_example = {
    'Operational Categorization Tier 1': ['Example Category'],
    'Summary': ['Example summary of the incident'],
    'Priority': ['High'],
    'Organization': ['IT'],
    'Department': ['Support']
}
predicted_group = predict(pd.DataFrame(input_example))
if predicted_group is not None:
    print(f'Predicted Assigned Group: {predicted_group.flatten()[0]}')
