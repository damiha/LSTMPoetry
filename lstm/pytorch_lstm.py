import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import pickle
import time
from tqdm import tqdm

class PytorchLSTMModel(nn.Module):
    
    def __init__(self, n_chars,  n_embed, block_size, hidden_size = 512, num_layers = 3, enable_cuda=True):
        
        super().__init__()
        
        # we add one additional row for the empty character (when data is missing)
        self.W = nn.Parameter(torch.randn(n_chars, n_embed))
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size = n_embed, hidden_size = self.hidden_size, num_layers = num_layers)
        self.hidden_to_output = nn.Linear(self.hidden_size, n_chars)
        
        self.block_size = block_size
        
        self.optim = None
        self.lr = None
        
        self.criterion = nn.CrossEntropyLoss()
        
        # for plotting the learning curves
        self.train_loss_per_epoch = []
        self.val_loss_per_epoch = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and enable_cuda else "cpu")
        print(f"device = {self.device}")
        
        self.to(self.device)
        
        
    def save_model(self, file_name):
        try:
            torch.save(self.state_dict(), file_name)
            print(f"Model successfully saved to {file_name}")
        except Exception as e:
            print(f"Error saving the model: {e}")

    def load_model(self, file_name):
        try:
            self.load_state_dict(torch.load(file_name, map_location=self.device))
            print(f"Model successfully loaded from {file_name}")
        except Exception as e:
            print(f"Error loading the model: {e}")
        
    def evaluate(self, X_val, y_val, batch_size):
            
            loss_per_batch = []
            
            N = len(X_val)
            
            with torch.no_grad():
            
                for batch_start in range(0, N, batch_size):
                    
                    batch_indices = torch.tensor(list(range(batch_start, min(N, batch_start + batch_size)))).to(torch.int)

                    batch = X_val[batch_indices].to(self.device)

                    logits = self.forward(batch)

                    true_labels = y_val[batch_indices].to(self.device)

                    logits = logits.transpose(1, 2)

                    loss = self.criterion(logits, true_labels)

                    loss_per_batch.append(loss.item())
                
            loss = np.mean(np.array(loss_per_batch))
            
            return loss
        
    def train(self, X_train, y_train, X_val, y_val, n_epochs, lr, batch_size = 64):
        
        if self.optim is None or self.lr is None or lr != self.lr:
            self.lr = lr
            self.optim = torch.optim.Adam(self.parameters(), lr=lr)
            
        N = len(X_train)
            
        for epoch_idx in range(n_epochs):
            
            perm = torch.randperm(N-1)
            
            loss_per_batch = []
            
            for batch_idx, batch_start in enumerate(tqdm(perm[::batch_size])):
                
                batch_indices = perm[batch_start:batch_start + batch_size]
                                
                batch = X_train[batch_indices].to(self.device)
                
                # logits = (batch size, seq length, n_chars)
                
                # a sequence of probability distributions
                
                logits = self.forward(batch)
                                
                true_labels = y_train[batch_indices].to(self.device)
                
                # nn.CrossEntropy needs [batch size, n_chars, seq_length]
                
                logits = logits.transpose(1, 2)
                
                # logits = (batch size, n_chars, seq length)
                
                loss = self.criterion(logits, true_labels)
                
                loss_per_batch.append(loss.item())
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            
            train_loss = np.mean(np.array(loss_per_batch))
            
            val_loss = self.evaluate(X_val, y_val, batch_size)
            
            self.train_loss_per_epoch.append(train_loss)
            self.val_loss_per_epoch.append(val_loss)
                
            print(f"{epoch_idx + 1}. train loss = {train_loss}, val loss = {val_loss}")
    
    def generate(self, context, n, temperature=1):
        
        # works with batches internally
        if context.ndim == 1:
            context = context.unsqueeze(dim=0)
        
        generated = []
        
        for _ in range(n):
            
            # TODO: add temperature sampling
            logits = self.forward(context)[0, -1] / temperature
            
            distribution = torch.distributions.Categorical(logits=logits)

            # Sample from the distribution
            sample = distribution.sample()
            
            generated.append(sample.item())

            context = torch.cat((context[0][1:], torch.tensor([sample]))).unsqueeze(0)
            
        return generated
    
    # only batches (do unsqueeze)
    def forward(self, batch):
        
        #print(f"Batch shape: {batch.shape}")
        
        # batch = (B, sequence_length)
        embeddings = self.W[batch]
        
        # embeddings = (B, sequence_length, n_embed)
        
        #print(f"Embedding shape: {embeddings.shape}")
        
        B = batch.shape[0]
        
        # nn.LSTM wants [sequence_length, batch_size, input_size]
        
        lstm_input = embeddings.transpose(0, 1)
        
        #print(f"LSTM input shape: {lstm_input.shape}")
        
        # lstm_out = what was thought at every new step ( we have seq length = 16, so 16 steps)
        lstm_out, (hidden, cell) = self.lstm(lstm_input)
        
        #print(f"LSTM output shape: {lstm_out.shape}")
                        
        last_hidden = lstm_out
        
        #print(f"Last hidden shape: {last_hidden.shape}")
                
        out = self.hidden_to_output(last_hidden).transpose(0, 1)
        
        #print(f"Final output shape: {out.shape}")
        
        return out