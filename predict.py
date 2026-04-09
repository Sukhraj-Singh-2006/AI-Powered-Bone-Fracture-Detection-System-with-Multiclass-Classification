import torch
from PIL import Image
from torchvision import transforms

from model import create_model


def predict(image_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # classes (same order as dataset folder)
    classes = ["fractured", "not fractured"
"avulsion fracture",
"comminuted fracture",
"compression fracture",
"fracture",
"greenstick fracture",
"hairline fracture",
"impacted fracture",
"intra-articular fracture",
"longitudinal fracture",
"oblique fracture",
"pathological fracture",
"spiral fracture"
]

    # image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    # load image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    # load model
    model = create_model(num_classes=2)
    model.load_state_dict(torch.load("models/best_model.pth"))
    model.to(device)

    model.eval()

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        _, predicted = torch.max(outputs,1)

    result = classes[predicted.item()]

    print("\nPrediction:", result)


if __name__ == "__main__":

    image_path = input("Enter image path: ")

    predict(image_path)