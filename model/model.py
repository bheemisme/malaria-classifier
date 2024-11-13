import torch
import torchvision.transforms as transforms

from model.malaria_classifier import MalariaClassifier

model = MalariaClassifier()

model.load_state_dict(torch.load('./model/malaria_classifier_epoch_50.pth',
                                 weights_only=False,
                                 map_location='cpu'))

class_names = ['Parasitized', 'Uninfected']


# Define the transform composition
transform = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256x256
    transforms.CenterCrop(224),  # Crop the center 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

def predict(image):
  image = transform(image).unsqueeze(0)  # Add batch dimension and move to device
  model.eval()

  with torch.no_grad():
    output = model(image)
    predicted_class = (output > 0.5).int().item()
  
  if predicted_class == 0:
    output = 1 - output
  return class_names[predicted_class], output.item()