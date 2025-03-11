import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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

# Function to get BERT embeddings
def get_bert_embeddings(texts, tokenizer, bert_model):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = bert_model(**inputs)
        # Get the mean of the last hidden state
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(embedding)
    # Convert list of arrays to a 2D array
    return np.vstack(embeddings)

# Load the dataset
data = pd.read_csv('incident_report_preprocessed_final_98000_cleaned.csv')

# Prepare input and output
X = data[['Operational Categorization Tier 1', 'Summary', 'Priority', 'Organization', 'Department']]
y = data['Assigned Group']

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X[['Operational Categorization Tier 1', 'Priority', 'Organization', 'Department']])

# Prepare BERT embeddings for the text features
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
summary_embeddings = get_bert_embeddings(X['Summary'].tolist(), tokenizer, bert_model)

# Combine the one-hot encoded features with BERT embeddings
X_final = np.hstack((X_encoded, summary_embeddings))

# Map y to numeric values
y_encoded = pd.factorize(y)[0]

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_final, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for batch processing
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
input_size = X_final.shape[1]
output_size = len(np.unique(y_encoded))
model = NeuralNetwork(input_size, output_size)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

# Validation phase
model.eval()
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    _, val_predicted = torch.max(val_outputs.data, 1)
    val_accuracy = accuracy_score(y_val, val_predicted)
    val_f1 = f1_score(y_val, val_predicted, average='weighted')
    val_precision = precision_score(y_val, val_predicted, average='weighted', zero_division=1)
    val_recall = recall_score(y_val, val_predicted, average='weighted', zero_division=1)
    print(f'Validation Metrics: Accuracy: {val_accuracy * 100:.2f}%, F1 Score: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')

# Testing phase
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, test_predicted = torch.max(test_outputs.data, 1)
    test_accuracy = accuracy_score(y_test, test_predicted)
    test_f1 = f1_score(y_test, test_predicted, average='weighted')
    test_precision = precision_score(y_test, test_predicted, average='weighted', zero_division=1)
    test_recall = recall_score(y_test, test_predicted, average='weighted', zero_division=1)
    print(f'Testing Metrics: Accuracy: {test_accuracy * 100:.2f}%, F1 Score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}')

# Save the trained model and preprocessors
def save_model_and_preprocessors(model, encoder, model_path='model_updated.pth', encoder_path='encoder_updated.pkl'):
    torch.save(model.state_dict(), model_path)
    joblib.dump(encoder, encoder_path)
    print("Model and preprocessors saved successfully.")

save_model_and_preprocessors(model, encoder)

# Reload the model and encoder for inference
def load_model_and_preprocessors(input_size, output_size, model_path='model_updated.pth', encoder_path='encoder_updated.pkl'):
    loaded_model = NeuralNetwork(input_size, output_size)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    loaded_encoder = joblib.load(encoder_path)
    return loaded_model, loaded_encoder

loaded_model, loaded_encoder = load_model_and_preprocessors(input_size, output_size)

# Prediction function for new inputs
def predict_with_saved_model(user_input):
    input_df = pd.DataFrame(user_input)
    input_encoded = loaded_encoder.transform(input_df[['Operational Categorization Tier 1', 'Priority', 'Organization', 'Department']])
    input_summary_embeddings = get_bert_embeddings(input_df['Summary'].tolist(), tokenizer, bert_model)
    # Ensure the input_final shape matches by stacking vertically
    input_final = np.hstack((input_encoded, input_summary_embeddings))
    input_tensor = torch.tensor(input_final, dtype=torch.float32)
    with torch.no_grad():
        output = loaded_model(input_tensor)
        _, predicted = torch.max(output.data, 1)
    return predicted.numpy()

# Example usage
user_input = {
    'Operational Categorization Tier 1': ['Failure'],
    'Summary': ['application working'],
    'Priority': ['Medium'],
    'Organization': ['CARGO'],
    'Department': ['CARGO OPERATIONS']
}
predicted_group = predict_with_saved_model(user_input)
print(f'Predicted Assigned Group: {predicted_group}')
