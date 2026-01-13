import matplotlib.pyplot as plt
from skimage import io, color, filters

from constants import RESOURCES_PATH

# Measuring the blood cells by means of an image segmentation by Amelia Carolina Sparavigna

# Ścieżka do obrazu
image_path = f"{RESOURCES_PATH}/images.jfif"

# Wczytanie obrazu
image = io.imread(image_path)

# Konwersja do skali szarości (jeśli obraz jest kolorowy)
if image.ndim == 3:
    gray_image = color.rgb2gray(image)
else:
    gray_image = image

# Wyznaczenie progu metodą Otsu
threshold_value = filters.threshold_otsu(gray_image)

# Progowanie (binaryzacja)
binary_image = gray_image > threshold_value

# Wizualizacja wyników
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(gray_image, cmap="gray")
axes[0].set_title("Obraz w skali szarości")
axes[0].axis("off")

axes[1].hist(gray_image.ravel(), bins=256)
axes[1].axvline(threshold_value, color='r', linestyle='--')
axes[1].set_title("Histogram i próg Otsu")

axes[2].imshow(binary_image, cmap="gray")
axes[2].set_title("Obraz po progowaniu (Otsu)")
axes[2].axis("off")

plt.tight_layout()
plt.show()
