from torch.distributions.normal import Normal
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

def croco(model,
          autoencoder,
          input_data,
          factual_class,
          weights,
          sigma2=0.3,
          max_iter=1000,
          learning_rate=0.01,
          device="cpu",
          n_samples=500):
    
    # check if weights are correct
    if weights is None:
        weights = {"robustness":1, "validity":1, "proximity":1}
    else:
        required_keys = ["robustness", "validity", "proximity"]
        assert all(key in weights for key in required_keys), "Dictionary is missing required keys"
    
    if autoencoder is None:
        perturbed_data, hist = alligator(
            model,
            input_data,
            factual_class,#
            weights,
            sigma2,
            max_iter,
            n_samples,
            learning_rate,
            device
        )
    else:
        perturbed_data, hist = alligator_latent(
            model,
            autoencoder,
            input_data,
            factual_class,#
            weights,
            sigma2,
            max_iter,
            n_samples,
            learning_rate,
            device
        )
    
    return perturbed_data, hist


def recourse_invalidation(model,input_data, perturbed_data, sigma2, device, n_samples):
    """
    Computes recourse invalidation rate around perturbed_data.
    
    """
    random_samples = reparametrization_trick_gaussian(perturbed_data, sigma2, device, n_samples)
    # Reapply normalization (added step)
    random_samples_normalized = transforms.Normalize((0.5,), (0.5,))(random_samples)
    validity = model(input_data)[:,factual_class] - (1 - model(random_samples_normalized)[:,factual_class])
    return torch.mean(torch.square(validity))

def recourse_invalidation_latent(model, autoencoder,input_data, perturbed_latent, sigma2, device, n_samples):
    """
    Computes recourse invalidation rate around perturbed_data in latent space.
    
    """
    random_samples = reparametrization_trick_gaussian(perturbed_latent, sigma2, device, n_samples)    
    #decode the latent representation
    random_samples = autoencoder.decode(random_samples)
    # Reapply normalization (added step)
    random_samples_normalized = transforms.Normalize((0.5,), (0.5,))(random_samples)
    validity = model(input_data)[:,factual_class] - (1 - model(random_samples_normalized)[:,factual_class])
    return torch.mean(torch.square(validity))


def alligator(model,
              input_data,
              factual_class,#
              weights,
              sigma2,
              max_iter,
              n_samples=500,
              learning_rate=0.01,
              device="cpu"):
    """
    Finds an adversarial perturbation for a PyTorch model in the latent space.

    Args:
        model (nn.Module): The PyTorch model to analyze.
        autoencoder (nn.Module): The autoencoder used to encode and decode the data.
        input_data (torch.Tensor): The initial datapoint.
        max_iterations (int): The maximum number of iterations to search.
        learning_rate (float, optional): The initial learning rate for Adam. Defaults to 0.001.
        device (str, optional): The device to use (CPU or GPU). Defaults to "cpu".

    Returns:
        torch.Tensor: The adversarial perturbation (if found), None otherwise.
    """

    # Initialize delta (perturbation in the latent space) and move it to the same device as the model
    delta = torch.zeros_like(input_data).to(device).requires_grad_()
    optimizer = optim.Adam([delta], lr=learning_rate)
    hist = {"loss":[]}

    # Perform optimization
    for epoch in range(max_iter):  # Adjust the number of epochs as needed
        optimizer.zero_grad()
        # Perturb the latent representation
        perturbed_data = input_data + delta
        # Reapply normalization (added step)
        perturbed_data_normalized = transforms.Normalize((0.5,), (0.5,))(perturbed_data)

        # Calculate loss based on the output label of your model
        robustness = weights["robustness"] * recourse_invalidation(model,input_data, perturbed_data, sigma2, device, n_samples)
        validity = weights["validity"] * torch.norm(model(input_data)[:,factual_class]-(1-model(perturbed_data_normalized)[:,factual_class]),p=2)
        proximity = weights["proximity"] * torch.norm(delta,p=1)
        #calculate proximity        
        loss =  robustness + validity + 0.05*proximity #lambda = 0.1

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Optional: print loss for monitoring
        hist["loss"].append(loss.item)

    return perturbed_data_normalized, hist


def alligator_latent(model,
                     autoencoder,
                     input_data,
                     factual_class,#
                     weights,
                     sigma2,
                     max_iter,
                     n_samples=500,
                     learning_rate=0.01,
                     device="cpu"):
    """
    Finds an adversarial perturbation for a PyTorch model in the latent space.

    Args:
        model (nn.Module): The PyTorch model to analyze.
        autoencoder (nn.Module): The autoencoder used to encode and decode the data.
        input_data (torch.Tensor): The initial datapoint.
        max_iterations (int): The maximum number of iterations to search.
        learning_rate (float, optional): The initial learning rate for Adam. Defaults to 0.001.
        device (str, optional): The device to use (CPU or GPU). Defaults to "cpu".

    Returns:
        torch.Tensor: The adversarial perturbation (if found), None otherwise.
    """

    # Initialize delta (perturbation in the latent space) and move it to the same device as the model
    latent_size = 4  # Adjust based on your latent space size
    delta = torch.zeros(1, latent_size, 7, 7).to(device).requires_grad_()
    optimizer = optim.Adam([delta], lr=learning_rate)
    hist = {"loss":[]}

    # Perform optimization
    for epoch in range(max_iter):  # Adjust the number of epochs as needed
        optimizer.zero_grad()
        # Get the reconstructed data and the latent representation
        latent_representation = autoencoder.encode(input_data)
        # Perturb the latent representation
        perturbed_latent = latent_representation + delta
        # Decode the perturbed latent space
        perturbed_data = autoencoder.decode(perturbed_latent)
        # Reapply normalization (added step)
        perturbed_data_normalized = transforms.Normalize((0.5,), (0.5,))(perturbed_data)

        # Calculate loss based on the output label of your model
        robustness = weights["robustness"] * recourse_invalidation_latent(
            model,
            autoencoder,
            input_data,
            perturbed_latent,
            sigma2,
            device,
            n_samples
        )
        validity = weights["validity"] * torch.norm(model(input_data)[:,0]-(1-model(perturbed_data_normalized)[:,0]),p=2)
        proximity = weights["proximity"] * torch.norm(delta,p=1)
        #calculate proximity        
        loss =  robustness + validity + 0.05*proximity #lambda=0.1

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Optional: print loss for monitoring
        hist["loss"].append(loss.item)

    # Check the final perturbed data and its label
    final_perturbed_data = autoencoder.decode(latent_representation + delta)
    final_output_label = model(final_perturbed_data)

    return final_perturbed_data, hist



def reparametrization_trick_gaussian(mu, sigma2, device, n_samples):
    # Expand the mean tensor to match the required shape
    mu_expanded = mu.expand(n_samples, -1, -1, -1)
    # Create a tensor for sigma2
    sigma2_tensor = torch.tensor(sigma2).to(device)
    # Create a Normal distribution with the given mean and standard deviation
    normal_dist = Normal(mu_expanded, torch.sqrt(sigma2_tensor))
    # Sample from the Normal distribution
    samples = normal_dist.sample()
    return samples