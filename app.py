from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

app = Flask(__name__)

# Load the model architecture
model = models.resnet50(pretrained=True)  # Use ResNet50
num_classes = 6  # Number of classes for your model

# Modify the final fully connected layer to match the number of classes
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the checkpoint, but filter out the 'fc' layer
checkpoint = torch.load('Trash_final_model.pth', map_location=torch.device('cpu'))

# Create a new state dict without the fc layer
filtered_checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith('fc')}

# Load the filtered weights into the model
model.load_state_dict(filtered_checkpoint, strict=False)

# Set the model to evaluation mode
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/classify', methods=['POST'])
def classify_image():
    # Check if an image was sent
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Process the image
    try:
        image = Image.open(file.stream).convert('RGB')  # Ensure the image is in RGB format
        image = transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Perform classification
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()

    # Class names
    class_names = ['BOTTLE', 'TIN', 'JUICE_BOX', 'MILK_CARTON', 'STYROFOAM', 'UTENSIL']
    predicted_class_name = class_names[predicted_class]

    return jsonify({'predicted_class': predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True)
