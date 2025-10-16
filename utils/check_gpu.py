import torch
print(torch.version.cuda)          # PyTorch CUDA version
print(torch.cuda.is_available())   # Should be True


from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet50(pretrained=False).to(device)

# Example for training loop:
for inputs, labels in dataloader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()