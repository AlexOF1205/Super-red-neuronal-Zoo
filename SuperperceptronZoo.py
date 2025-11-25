import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import zipfile

# ================================
# 1. Cargar dataset Zoo desde zoo.zip
# ================================
zip_path = "zoo.zip"  

with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open('zoo.data') as f:
        df = pd.read_csv(f, header=None)

# ================================
# 2. Agregar nombres de columnas
# ================================
columnas = [
    "animal_name",
    "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator",
    "toothed", "backbone", "breathes", "venomous", "fins",
    "legs", "tail", "domestic", "catsize", "class"
]
df.columns = columnas


X = df.drop(columns=["animal_name", "class"])
y = df["class"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# ================================
# 3. Train y test split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ================================
# 4. Tensores
# ================================
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# ================================
# 5. Modelo
# ================================
model = nn.Sequential(
    nn.Linear(16, 32),          #Capa oculta con 32 neuronas
    nn.ReLU(),
    nn.Linear(32, 7)            #Capa de salida con 7 clases
)

# ================================
# 6. Entrenamiento
# ================================
criterio = nn.CrossEntropyLoss()                #Función de pérdida para clasificación múltiple
optimizer = optim.SGD(model.parameters(), lr=0.1) #Taza de aprendizaje de 0.1

for epoca in range(1000):                          #1000 épocas
    pred = model(X_train_tensor)  
    perdida = criterio(pred, y_train_tensor)

    optimizer.zero_grad()
    perdida.backward()
    optimizer.step()

# ================================
# 7. Predicción
# ================================
with torch.no_grad():
    pred_test = model(X_test_tensor)
    pred_labels = torch.argmax(pred_test, dim=1)
    accuracy = (pred_labels == y_test_tensor).float().mean()
    print("Accuracy:", accuracy.item())
