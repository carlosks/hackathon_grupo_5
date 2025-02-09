import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
import time

# 🚀 1️⃣ Configurações do Treinamento
DATASET_PATH = "/Users/carlosks/Desktop/Deputado/reconhecimento_facial_webcam/Datasets"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🚀 2️⃣ Transformações para Pré-processamento das Imagens
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar para MobileNetV2
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 🚀 3️⃣ Carregar Dataset de Imagens
train_dataset = datasets.ImageFolder(root=f"{DATASET_PATH}/train", transform=transform)
val_dataset = datasets.ImageFolder(root=f"{DATASET_PATH}/val", transform=transform)

# Verificar tamanho do dataset
print(f"🔍 Tamanho do dataset de treino: {len(train_dataset)} imagens")
print(f"🔍 Tamanho do dataset de validação: {len(val_dataset)} imagens")

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("❌ ERRO: O dataset de treino ou validação está vazio!")

# Criar DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 🚀 4️⃣ Carregar MobileNetV2 Pré-treinado
model = models.mobilenet_v2(weights="IMAGENET1K_V1")  # Carregar pesos treinados no ImageNet

# Substituir a última camada para classificar "facas" e "não facas"
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)  # Duas classes: faca (1), não faca (0)

print(f"Classes do dataset: {train_dataset.class_to_idx}")

# Enviar para GPU se disponível
model = model.to(DEVICE)

# 🚀 5️⃣ Definir Função de Perda e Otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 🚀 6️⃣ Loop de Treinamento
def train(model, train_loader, val_loader, epochs):
    best_accuracy = 0.0
    for epoch in range(epochs):
        print(f"🚀 Iniciando Epoch {epoch+1}/{epochs}")
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 10 == 0:
                print(f"🟢 Batch {batch_idx}/{len(train_loader)} - Perda: {loss.item():.4f}")

        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # 🚀 7️⃣ Validação do Modelo
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = 100 * val_correct / val_total

        # 🔹 Salvar Melhor Modelo
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "mobilenetv2_faca_vs_nao_faca.pth")

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"🔹 Epoch [{epoch+1}/{epochs}], Tempo: {epoch_time:.2f}s, Perda: {train_loss:.4f}, Acurácia Treino: {train_accuracy:.2f}%, Acurácia Validação: {val_accuracy:.2f}%")

# 🚀 8️⃣ Executar Treinamento
train(model, train_loader, val_loader, EPOCHS)

print("✅ Treinamento Concluído!")