import matplotlib.pyplot as plt
import numpy as np

# Correction : définir epochs pour correspondre à la longueur des autres listes
epochs = range(1, 11)  # Maintenant de 1 à 10 au lieu de 1 à 100

train_acc = [0.8734, 0.9256, 0.9478, 0.9623, 0.9778, 0.9845, 0.9889, 0.9923, 0.9956, 0.9978]
val_acc = [0.9012, 0.9234, 0.9345, 0.9412, 0.9445, 0.9501, 0.9567, 0.9589, 0.9623, 0.9645]
train_loss = [0.3587, 0.2145, 0.1523, 0.1102, 0.0823, 0.0634, 0.0456, 0.0321, 0.0234, 0.0189]
val_loss = [0.3054, 0.2567, 0.2234, 0.2101, 0.1987, 0.1823, 0.1676, 0.1612, 0.1554, 0.1498]

# Données sur la distribution des images par classe
flower_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
image_counts = [764, 1052, 784, 733, 984]

# 1. Graphique de l'accuracy d'entraînement et de validation
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 2. Graphique de la perte d'entraînement et de validation
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 3. Distribution des images par classe
plt.figure(figsize=(10, 6))
plt.bar(flower_names, image_counts)
plt.title('Distribution of Images per Flower Class')
plt.xlabel('Flower Types')
plt.ylabel('Number of Images')
plt.show()

# 4. Matrice de confusion (exemple hypothétique)
conf_matrix = np.array([
    [145, 3, 1, 0, 1],
    [2, 208, 0, 1, 1],
    [1, 0, 155, 2, 2],
    [0, 1, 3, 144, 0],
    [2, 1, 1, 0, 196]
])

plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(flower_names))
plt.xticks(tick_marks, flower_names, rotation=45)
plt.yticks(tick_marks, flower_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
# Ajout des valeurs dans les cellules
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")
plt.tight_layout()
plt.show()