# from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# import joblib
# import datetime
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import pandas as pd
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import OneHotEncoder
# from transformers import BertTokenizer, BertModel
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import joblib
# import datetime

# # Define the modified neural network model with BERT fine-tuning
# class NeuralNetworkWithBERT(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(NeuralNetworkWithBERT, self).__init__()
#         self.fc1 = nn.Linear(input_size, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.fc3 = nn.Linear(256, 128)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.fc4 = nn.Linear(128, output_size)
#         self.leaky_relu = nn.LeakyReLU(0.01)
#         self.dropout = nn.Dropout(0.3)
#         self.softmax = nn.LogSoftmax(dim=1)
        
#         # BERT model initialization
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.bert_dropout = nn.Dropout(0.3)

#     def forward(self, x, input_ids, attention_mask):
#         # BERT output
#         bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         bert_pooler_output = bert_outputs.pooler_output  # Pool the last hidden state
        
#         # Combine BERT features with input features
#         x = torch.cat((x, bert_pooler_output), dim=1)

#         # Pass through fully connected layers
#         x = self.leaky_relu(self.bn1(self.fc1(x)))
#         x = self.dropout(x)
#         x = self.leaky_relu(self.bn2(self.fc2(x)))
#         x = self.dropout(x)
#         x = self.leaky_relu(self.bn3(self.fc3(x)))
#         x = self.fc4(x)
#         return self.softmax(x)

# # Function to get BERT embeddings
# def get_bert_embeddings(texts, tokenizer, bert_model):
#     input_ids = []
#     attention_masks = []
    
#     for text in texts:
#         encoding = tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             padding='max_length',
#             max_length=512,
#             return_attention_mask=True,
#             return_tensors='pt',
#             truncation=True
#         )
#         input_ids.append(encoding['input_ids'])
#         attention_masks.append(encoding['attention_mask'])
        
#     input_ids = torch.cat(input_ids, dim=0)
#     attention_masks = torch.cat(attention_masks, dim=0)
    
#     return input_ids, attention_masks

# # Get current timestamp
# timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# # Load the dataset (Make sure the dataset is uploaded to Kaggle's environment)
# data = pd.read_csv('incident_report_preprocessed_final_98000_cleaned.csv')  # Replace with actual dataset path

# # Prepare input and output
# X = data[['Operational Categorization Tier 1', 'Summary', 'Priority', 'Organization', 'Department']]
# y = data['Assigned Group']

# # One-hot encode categorical features
# encoder = OneHotEncoder(sparse_output=False)
# X_encoded = encoder.fit_transform(X[['Operational Categorization Tier 1', 'Priority', 'Organization', 'Department']])

# # Prepare BERT embeddings for the text features
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# input_ids, attention_masks = get_bert_embeddings(X['Summary'].tolist(), tokenizer, None)

# # Combine the one-hot encoded features with BERT embeddings
# X_final = np.hstack((X_encoded, input_ids.numpy()))

# # Map y to numeric values and save the mapping
# y_encoded, y_mapping = pd.factorize(y)
# y_mapping_dict = dict(enumerate(y_mapping))  # Create mapping for y

# # Create an output folder in Kaggle to save models and results
# output_dir = f"model_output_{timestamp}"
# os.makedirs(output_dir, exist_ok=True)

# joblib.dump(y_mapping_dict, f'{output_dir}_new_y_mapping.pkl')  # Save y mapping

# # Save the mapping for X (One-Hot Encoded features)
# X_mapping_dict = {i: col for i, col in enumerate(encoder.get_feature_names_out())}  # Create mapping for X
# joblib.dump(X_mapping_dict, f'{output_dir}_new_X_mapping.pkl')  # Save X mapping

# # Split the data into training, validation, and testing sets
# X_train, X_temp, y_train, y_temp = train_test_split(X_final, y_encoded, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# # Save X_test and y_test to separate files for future use
# np.save(f'{output_dir}_X_test.npy', X_test)
# np.save(f'{output_dir}_y_test.npy', y_test)
# np.save(f'{output_dir}_X_val.npy', X_val)
# np.save(f'{output_dir}_y_val.npy', y_val)

# # Convert to PyTorch tensors
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
# X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
# y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
# y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
# input_ids_train = input_ids[:len(X_train_tensor)].to(device)
# attention_masks_train = attention_masks[:len(X_train_tensor)].to(device)

# # Create DataLoader for batch processing
# batch_size = 64
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor, input_ids_train, attention_masks_train)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # Initialize the model, loss function, and optimizer
# input_size = X_final.shape[1]
# output_size = len(np.unique(y_encoded))
# model = NeuralNetworkWithBERT(input_size, output_size).to(device)
# criterion = nn.NLLLoss()
# optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# # Learning Rate Scheduler
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*100)

# # Training loop with Early Stopping
# epochs = 100
# best_val_accuracy = 0
# patience = 10  # Early stopping patience
# counter = 0

# for epoch in range(epochs):
#     model.train()
#     epoch_loss = 0
#     for batch_X, batch_y, batch_input_ids, batch_attention_mask in train_loader:
#         optimizer.zero_grad()
#         outputs = model(batch_X, batch_input_ids, batch_attention_mask)
#         loss = criterion(outputs, batch_y)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
    
#     scheduler.step()
#     print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}')
    
#     # Validation metrics
#     model.eval()
#     with torch.no_grad():
#         val_outputs = model(X_val_tensor, input_ids[:len(X_val_tensor)].to(device), attention_masks[:len(X_val_tensor)].to(device))
#         _, val_predicted = torch.max(val_outputs.data, 1)
#         val_accuracy = accuracy_score(y_val, val_predicted.cpu().numpy())
#         print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
        
#         if val_accuracy > best_val_accuracy:
#             best_val_accuracy = val_accuracy
#             counter = 0
#             # Save the best model
#             torch.save(model.state_dict(), f'{output_dir}_best_trained_model_{timestamp}.pth')
#         else:
#             counter += 1
#             if counter >= patience:
#                 print("Early stopping due to no improvement in validation accuracy.")
#                 break

# # Test metrics
# with torch.no_grad():
#     test_outputs = model(X_test_tensor, input_ids[:len(X_test_tensor)].to(device), attention_masks[:len(X_test_tensor)].to(device))
#     _, test_predicted = torch.max(test_outputs.data, 1)
#     test_accuracy = accuracy_score(y_test, test_predicted.cpu().numpy())
#     print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# # Save the encoder and metadata
# joblib.dump(encoder, f'{output_dir}_new_trained_encoder.pkl')
# joblib.dump({'input_size': input_size, 'output_size': output_size}, f'{output_dir}_new_metadata.pkl')
# print("Model, encoder, and metadata saved successfully.")


import logging
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the modified neural network model with BERT fine-tuning
class NeuralNetworkWithBERT(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetworkWithBERT, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, output_size)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.LogSoftmax(dim=1)
        
        # BERT model initialization
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_dropout = nn.Dropout(0.3)

    def forward(self, x, input_ids, attention_mask):
        # BERT output
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_pooler_output = bert_outputs.pooler_output  # Pool the last hidden state
        
        # Combine BERT features with input features
        x = torch.cat((x, bert_pooler_output), dim=1)

        # Pass through fully connected layers
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return self.softmax(x)

# Function to get BERT embeddings
def get_bert_embeddings(texts, tokenizer):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
        
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    return input_ids, attention_masks

# Get current timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Load the dataset
logger.debug("Loading dataset...")
data = pd.read_csv('incident_report_preprocessed_final_98000_cleaned.csv')  # Replace with actual dataset path
logger.debug(f"Dataset loaded with {len(data)} rows.")

# Prepare input and output
X = data[['Operational Categorization Tier 1', 'Summary', 'Priority', 'Organization', 'Department']]
y = data['Assigned Group']
logger.debug("Input and output prepared.")

# One-hot encode categorical features
logger.debug("One-hot encoding categorical features...")
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X[['Operational Categorization Tier 1', 'Priority', 'Organization', 'Department']])
logger.debug(f"One-hot encoded features shape: {X_encoded.shape}")

# Prepare BERT embeddings for the text features
logger.debug("Generating BERT embeddings...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input_ids, attention_masks = get_bert_embeddings(X['Summary'].tolist(), tokenizer)
logger.debug(f"BERT embeddings generated. input_ids shape: {input_ids.shape}, attention_masks shape: {attention_masks.shape}")

# Combine the one-hot encoded features with BERT embeddings
X_final = np.hstack((X_encoded, input_ids.numpy()))
logger.debug(f"Combined features shape: {X_final.shape}")

# Map y to numeric values and save the mapping
y_encoded, y_mapping = pd.factorize(y)
y_mapping_dict = dict(enumerate(y_mapping))  # Create mapping for y
logger.debug(f"y encoded with {len(y_mapping)} unique classes.")

# Create an output folder to save models and results
output_dir = f"model_output_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
logger.debug(f"Output directory created: {output_dir}")

# Save y mapping inside output_dir
joblib.dump(y_mapping_dict, os.path.join(output_dir, 'new_y_mapping.pkl'))
logger.debug("y mapping saved.")

# Save X mapping inside output_dir
X_mapping_dict = {i: col for i, col in enumerate(encoder.get_feature_names_out())}  # Create mapping for X
joblib.dump(X_mapping_dict, os.path.join(output_dir, 'new_X_mapping.pkl'))
logger.debug("X mapping saved.")

# Split the data into training, validation, and testing sets
logger.debug("Splitting data into training, validation, and testing sets...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
logger.debug(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}, Test set size: {len(X_test)}")

# Now X_temp is a DataFrame, so you can access its index
val_texts = X_temp.iloc[:len(X_val)]['Summary'].tolist()  # Texts for validation set
test_texts = X_temp.iloc[len(X_val):]['Summary'].tolist()  # Texts for test set
logger.debug(f"Validation texts: {len(val_texts)}, Test texts: {len(test_texts)}")

# Generate BERT embeddings for validation and test sets
logger.debug("Generating BERT embeddings for validation and test sets...")
input_ids_val, attention_masks_val = get_bert_embeddings(val_texts, tokenizer)
input_ids_test, attention_masks_test = get_bert_embeddings(test_texts, tokenizer)
logger.debug(f"Validation BERT embeddings: input_ids_val shape: {input_ids_val.shape}, attention_masks_val shape: {attention_masks_val.shape}")
logger.debug(f"Test BERT embeddings: input_ids_test shape: {input_ids_test.shape}, attention_masks_test shape: {attention_masks_test.shape}")

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(f"Using device: {device}")

X_train_tensor = torch.tensor(X_encoded[:len(X_train)], dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_encoded[len(X_train):len(X_train) + len(X_val)], dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_encoded[len(X_train) + len(X_val):], dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

input_ids_train = input_ids[:len(X_train_tensor)].to(device)
attention_masks_train = attention_masks[:len(X_train_tensor)].to(device)
input_ids_val = input_ids_val.to(device)
attention_masks_val = attention_masks_val.to(device)
input_ids_test = input_ids_test.to(device)
attention_masks_test = attention_masks_test.to(device)

logger.debug(f"X_train_tensor shape: {X_train_tensor.shape}, y_train_tensor shape: {y_train_tensor.shape}")
logger.debug(f"X_val_tensor shape: {X_val_tensor.shape}, y_val_tensor shape: {y_val_tensor.shape}")
logger.debug(f"X_test_tensor shape: {X_test_tensor.shape}, y_test_tensor shape: {y_test_tensor.shape}")
logger.debug(f"input_ids_train shape: {input_ids_train.shape}, attention_masks_train shape: {attention_masks_train.shape}")
logger.debug(f"input_ids_val shape: {input_ids_val.shape}, attention_masks_val shape: {attention_masks_val.shape}")
logger.debug(f"input_ids_test shape: {input_ids_test.shape}, attention_masks_test shape: {attention_masks_test.shape}")

# Create DataLoader for batch processing
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor, input_ids_train, attention_masks_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
logger.debug(f"DataLoader created with batch size: {batch_size}")

# Initialize the model, loss function, and optimizer
input_size = X_final.shape[1]
output_size = len(np.unique(y_encoded))
logger.debug(f"Model input size: {input_size}, output size: {output_size}")

model = NeuralNetworkWithBERT(input_size, output_size).to(device)
criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
logger.debug("Model, loss function, and optimizer initialized.")

# Learning Rate Scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*100)
logger.debug("Learning rate scheduler initialized.")

# Training loop with Early Stopping
epochs = 100
best_val_loss = float('inf')
patience = 10  # Early stopping patience
counter = 0

logger.debug("Starting training loop...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y, batch_input_ids, batch_attention_mask in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X, batch_input_ids, batch_attention_mask)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    logger.debug(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}')
    
    # Validation metrics
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor, input_ids_val, attention_masks_val)
        val_loss = criterion(val_outputs, y_val_tensor)
        _, val_predicted = torch.max(val_outputs.data, 1)
        val_accuracy = accuracy_score(y_val, val_predicted.cpu().numpy())
        logger.debug(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save the best model
            model_path = os.path.join(output_dir, f'best_trained_model_{timestamp}.pth')
            torch.save(model.state_dict(), model_path)
            logger.debug(f"Best model saved at: {model_path}")
        else:
            counter += 1
            if counter >= patience:
                logger.debug("Early stopping due to no improvement in validation loss.")
                break

# Test metrics
logger.debug("Evaluating on test set...")
model_path = os.path.join(output_dir, f'best_trained_model_{timestamp}.pth')
model.load_state_dict(torch.load(model_path))
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor, input_ids_test, attention_masks_test)
    _, test_predicted = torch.max(test_outputs.data, 1)
    test_accuracy = accuracy_score(y_test, test_predicted.cpu().numpy())
    logger.debug(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Save the encoder and metadata
encoder_path = os.path.join(output_dir, 'new_trained_encoder.pkl')
joblib.dump(encoder, encoder_path)
metadata_path = os.path.join(output_dir, 'new_metadata.pkl')
joblib.dump({'input_size': input_size, 'output_size': output_size}, metadata_path)
logger.debug("Model, encoder, and metadata saved successfully.")