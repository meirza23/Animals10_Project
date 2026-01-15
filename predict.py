import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import argparse
from src.model import AnimalCNN

# Sınıf isimleri (klasör yapısından alındı)
CLASS_NAMES = [
    'butterfly', 'cat', 'chicken', 'cow', 'dog', 
    'elephant', 'horse', 'sheep', 'spider', 'squirrel'
]

IMG_SIZE = 224
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(model_path, device):
    """Eğitilmiş modeli yükler."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    
    model = AnimalCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device):
    """Verilen resim için tahmin yapar."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Resim dosyası bulunamadı: {image_path}")

    # Resim ön işleme
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.topk(probabilities, 1)
        
        predicted_class = CLASS_NAMES[top_class.item()]
        confidence = top_prob.item() * 100

    return predicted_class, confidence

def main():
    parser = argparse.ArgumentParser(description="AnimalCNN ile resim tahmini yap.")
    parser.add_argument("image_path", type=str, help="Tahmin edilecek resmin yolu")
    parser.add_argument("--model", type=str, default="model.pth", help="Model dosyasının yolu (varsayılan: model.pth)")
    
    args = parser.parse_args()
    
    try:
        print(f"Model yükleniyor: {args.model}...")
        model = load_trained_model(args.model, DEVICE)
        
        print(f"Tahmin ediliyor: {args.image_path}...")
        predicted_class, confidence = predict_image(model, args.image_path, DEVICE)
        
        print("-" * 30)
        print(f"Tahmin: {predicted_class}")
        print(f"Güven: %{confidence:.2f}")
        print("-" * 30)
        
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    main()
