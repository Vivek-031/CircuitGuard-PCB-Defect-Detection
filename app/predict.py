import torch
import timm
from PIL import Image
from torchvision import transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CLASSES = [
    'Missing_hole',
    'Mouse_bite',
    'Open_circuit',
    'Short',
    'Spur',
    'Spurious_copper'
]

MODEL_PATH = "app/model/circuitguard_efficientnet_b4_v2.pth"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

@torch.no_grad()
def load_model():

    model = timm.create_model(
        'efficientnet_b4',
        pretrained=False,
        num_classes=len(CLASSES)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint['model'])

    model.to(DEVICE)
    model.eval()

    return model

model = load_model()

def predict_image(image):

    image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    outputs = model(tensor)

    probs = torch.softmax(outputs, dim=1)

    confidence, pred = torch.max(probs, dim=1)

    predicted_class = CLASSES[pred.item()]

    confidence_score = confidence.item() * 100

    return predicted_class, confidence_score
