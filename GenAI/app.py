from flask import Flask, request, send_file
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os


app = Flask(__name__)


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def denormalize(tensor):
    """Convert output tensor back to normal image"""
    tensor = tensor.detach().cpu()
    tensor = (tensor * 0.5) + 0.5
    tensor = torch.clamp(tensor, 0, 1)
    img = transforms.ToPILImage()(tensor.squeeze())
    return img


# Simple Generator class (from your notebook)
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=6):
        super(Generator, self).__init__()

        model = [
            nn.Conv2d(input_nc, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        ]

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(256)]

        model += [
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, output_nc, 7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
ghibli_model = None
sketch_model = None

def load_models():
    global ghibli_model, sketch_model
    
    try:
        # Try to load Ghibli model
        ghibli_model = Generator().to(device)
        if os.path.exists("generator_ghibli.pth"):
            ghibli_model.load_state_dict(torch.load("generator_ghibli.pth", map_location=device))
            print("Loaded Ghibli model")
        elif os.path.exists("../checkpoints/G_AB.pth"):
            ghibli_model.load_state_dict(torch.load("../checkpoints/G_AB.pth", map_location=device))
            print("Loaded Ghibli model from checkpoints")
        else:
            print("No Ghibli model found, using random weights")
        ghibli_model.eval()
        
        # Try to load Sketch model
        sketch_model = Generator().to(device)
        if os.path.exists("generator_sketch_1.pth"):
            sketch_model.load_state_dict(torch.load("generator_sketch_1.pth", map_location=device))
            print("Loaded Sketch model")
        else:
            sketch_model = ghibli_model  # Use same model for both if sketch not available
            print("No Sketch model found, using Ghibli model")
        sketch_model.eval()
        
    except Exception as e:
        print(f"Error loading models: {e}")
        # Create dummy models that just return the input
        ghibli_model = Generator().to(device)
        sketch_model = Generator().to(device)


def apply_model(img, model_type):
    if ghibli_model is None or sketch_model is None:
        # If models aren't loaded, just resize and return the input image
        return img.resize((256, 256))
    
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_type == "ghibli":
            output = ghibli_model(img_tensor)
        elif model_type == "sketch":
            output = sketch_model(img_tensor)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    output_img = denormalize(output)
    return output_img


@app.route('/')
def index():
    return send_file('index.html')


@app.route('/process', methods=['POST'])
def process():
    file = request.files["image"]

   
    model_type = request.form.get("model_type")

    if model_type not in ["ghibli", "sketch"]:
        return {"error": "Invalid model type received"}, 400

    img = Image.open(file.stream).convert("RGB")
    output_img = apply_model(img, model_type)

    buffer = io.BytesIO()
    output_img.save(buffer, format="PNG")
    buffer.seek(0)

    return send_file(buffer, mimetype="image/png")


# ---------------------------------------
if __name__ == '__main__':
    print("Starting GenAI Style Transfer App...")
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
