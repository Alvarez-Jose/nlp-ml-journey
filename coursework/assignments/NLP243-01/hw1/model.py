"""
STUDENT IMPLEMENTATION REQUIRED

This file contains the model architecture that you need to implement for HW1.
You should complete the BoWClassifier class by implementing the forward method
and any other necessary components.
"""

import torch
from torch import nn

class BoWClassifier(nn.Module):
    def __init__(self, input_size, num_labels):
        super().__init__()
        self.num_labels = num_labels
        # Hidden MLP head for capacity
        self.ff1 = nn.Linear(input_size, 256) # hidden layer - adjust addcordingly 
        self.ff2 = nn.Linear(256, num_labels)
        # can keep ff3 - choose to comment it out. could be useful 
        #self.ff3 = nn.Linear(input_size, num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.ff1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ff2(x)
        return x
        
    
    
def get_best_model(input_size, num_labels):
    # return a newly instantiated model that your best weights will be loaded into
    # the model returned by this function must exactly match the architecture that the saved weights expect
    return BoWClassifier(input_size=input_size, num_labels=num_labels)

def predict(model_output):
    """
    Converts model output to class predictions.
    Args:
        model_output: Output from model.forward(x)
    Returns:
        predictions: Tensor of predicted class labels
    """
    # Slightly relaxed threshold helps recall for rare labels
    return (torch.sigmoid(model_output) >= 0.30).int()

