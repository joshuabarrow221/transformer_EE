"""
define loss function
"""

import torch

# shape of output and target: (batch_size, output_dim)
# shape of weight: (batch_size, 1)

# for shifting
constant = 2.0

# base loss functions
def MSE_loss(output, target, weight=None):
    """
    mean squared error loss
    """
    # Note: torch.mean() returns the mean value of all elements in the input tensor, which is a scalar value.
    if weight is None:
        return torch.mean((output - target) ** 2)
    return torch.mean(weight * (output - target) ** 2)

def MACE_loss(output, target, weight=None):
    """
    mean absolute cubic error loss
    """
    # Note: torch.mean() returns the mean value of all elements in the input tensor, which is a scalar value.
    if weight is None:
        return torch.mean(torch.abs(output - target) ** 3)
    return torch.mean(weight * torch.abs(output - target) ** 3)


def MAE_loss(output, target, weight=None):
    """
    mean absolute error loss
    """
    if weight is None:
        return torch.mean(torch.abs(output - target))
    return torch.mean(weight * torch.abs(output - target))


def MAPE_loss(output, target, weight=None):
    """
    mean absolute percentage error loss, similar to L1 loss
    """
    if weight is None:
        return torch.mean(torch.abs(output - target) / (0.01 + torch.abs(target)))
    return torch.mean(weight * torch.abs(output - target) / (0.01 + torch.abs(target)))


##################################
# shifted loss functions
#################################

def MSE_loss_shifted(output, target, weight=None):
    """
    shifted mean squared error loss
    """
    # Note: torch.mean() returns the mean value of all elements in the input tensor, which is a scalar value.
    if weight is None:
        return torch.mean((( output + constant ) - ( target )) ** 2)
    return torch.mean(weight * (( output + constant ) - ( target )) ** 2)

def MACE_loss_shifted(output, target, weight=None):
    """
    shifted mean absolute cubic error loss
    """
    # Note: torch.mean() returns the mean value of all elements in the input tensor, which is a scalar value.
    if weight is None:
        return torch.mean(torch.abs(( output + constant ) - ( target )) ** 3)
    return torch.mean(weight * torch.abs(( output + constant ) - ( target )) ** 3)


def MAE_loss_shifted(output, target, weight=None):
    """
    shifted mean absolute error loss
    """
    if weight is None:
        return torch.mean(torch.abs(( output - constant ) - ( target )))
    return torch.mean(weight * torch.abs(( output - constant ) - ( target )))


def MAPE_loss_shifted(output, target, weight=None):
    """
    shifted mean absolute percentage error loss, similar to L1 loss
    """
    if weight is None:
        return torch.mean(torch.abs( ( output + constant) - ( target )) / (0.01 + torch.abs( target )))
    return torch.mean(weight * torch.abs( ( output + constant) - ( target )) / (0.01 + torch.abs( target )))


# add base loss functions to loss_function dictionary
loss_function = {
    "mean squared error": MSE_loss,
    "mean absolute error": MAE_loss,
    "mean absolute percentage error": MAPE_loss,
    "mean absolute cubic error": MACE_loss,
    "shifted mean squared error": MSE_loss_shifted,
    "shifted mean absolute error": MAE_loss_shifted,
    "shifted mean absolute percentage error": MAPE_loss_shifted,
    "shifted mean absolute cubic error": MACE_loss_shifted
}


# set up a base loss function by name
def get_loss_function(loss_function_name, output, target, weight=None):
    """
    get loss function
    """
    return loss_function[loss_function_name](output, target, weight)


# complex loss functions utilize base loss functions


def linear_combination_loss(output, target, weight=None, **kwargs):
    """
    linear combination of base loss functions
    coefficients, base_loss_names should have the same length, which is the number of output variables
    e.g. kwargs = {"coefficients": [0.5, 0.5], "base_loss_names": ["mean squared error", "mean absolute error"]}
    """
    if "base_loss_names" not in kwargs or "coefficients" not in kwargs:
        raise ValueError("base_loss_names and coefficients must be provided in kwargs")

    if len(kwargs["base_loss_names"]) != len(kwargs["coefficients"]):
        raise ValueError(
            "base_loss_names and coefficients must have the same length\n",
            "len(base_loss_names):",
            len(kwargs["base_loss_names"]),
            "\nlen(coefficients):",
            len(kwargs["coefficients"]),
        )

    base_loss_names = kwargs["base_loss_names"]
    coefficients = kwargs["coefficients"]
    linear_loss = 0
    for i in range(len(base_loss_names)):
        linear_loss += coefficients[i] * loss_function[base_loss_names[i]](
            output[:, i], target[:, i], torch.squeeze(weight)
        )
    return linear_loss

def abs_invariant_mass_squared_prediction_loss(output, weight=None):
    """
    A function to calculate the invariant mass prediction of the neutrino from the output.
    The function should be used in concert with the linear_combination_loss function
    as a source of "continuous pretraining" to seed the model outputs with a good starting point.
    Assumes the output is a 4-vector with the first element being the energy and the last three elements being the momentum
    """
    
    #abs_inv_mass_sq = 0 #initialize the loss to a nonphysical value--necessary?
    #Assumes an ordering to the output, which is: energy, px, py, pz
    #pred_energy = output[:, 0].item()
    
    # if weight is None:
    #     return torch.mean((output[:, 0]**2 - output[:, 1]**2 - output[:, 2]**2 - output[:, 3]**2))**2
    # return torch.mean(weight * (output[:, 0]**2 - output[:, 1]**2 - output[:, 2]**2 - output[:, 3]**2))**2
    if weight is None:
        return 125.0*torch.mean((output[:, 0]**2 - output[:, 1]**2 - output[:, 2]**2 - output[:, 3]**2))**2
    return 125.0*torch.mean(weight * (output[:, 0]**2 - output[:, 1]**2 - output[:, 2]**2 - output[:, 3]**2))**2

def abs_invariant_mass_squared_prediction_loss_shifted(output, weight=None):
    """
    A function to calculate the invariant mass prediction of the neutrino from the output.
    The function should be used in concert with the linear_combination_loss function
    as a source of "continuous pretraining" to seed the model outputs with a good starting point.
    Assumes the output is a 4-vector with the first element being the energy and the last three elements being the momentum
    """
    
    #abs_inv_mass_sq = 0 #initialize the loss to a nonphysical value--necessary?
    #Assumes an ordering to the output, which is: energy, px, py, pz
    #pred_energy = output[:, 0].item()
    
    if weight is None:
        return torch.mean(((output[:, 0] + constant)**2 - (output[:, 1] + constant)**2 - (output[:, 2] + constant)**2 - (output[:, 3] + constant)**2))**2
    return torch.mean(weight * ((output[:, 0] + constant)**2 - (output[:, 1] + constant)**2 - (output[:, 2] + constant)**2 - (output[:, 3] + constant)**2))**2
    #     return torch.mean(torch.abs(output - target))
    # return torch.mean(weight * torch.abs(output - target))
