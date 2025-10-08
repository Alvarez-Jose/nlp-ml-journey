"""
STUDENT IMPLEMENTATION REQUIRED

This file contains the model architecture that you need to implement for HW1.
You should complete the BoWClassifier class by implementing the forward method
and any other necessary components.
"""

import torch
from torch import nn


# defines the exact same model architecture
class BoWClassifier(nn.Module):
    def __init__(self, input_size, num_labels):
        super().__init__()
        self.num_labels = num_labels

    
    def forward(self, x):
        return torch.zeros((x.shape[0], self.num_labels), device=x.device)
    
    
def get_best_model(self, input_size, num_labels):
    # return a newly instantiated model that your best weights will be loaded into

    # Define the path to your saved state dictionary
    weights_path = 'best_model.pt'

    # Load the saved state dictionary
    state_dict = torch.load(weights_path)

    # Load the weights into the newly instanitated model
    get_best_model.load_state_dict(state_dict)
    get_best_model.eval()
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
    if model_output() < 0 or model_output.max() > 1:
        probabilites = torch.sigmoid(model_output)
    else:
        probabilites = model_output

    model_output = (probabilites >= model_output).init()
    


    return model_output > 0

