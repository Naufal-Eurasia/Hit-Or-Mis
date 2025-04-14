# Hit-Or-Mis
Menguji strel Hit-Or-Mis,Binary,Complement Image,Skeletonization,Thinning
```python
# ✅ Import library
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import invert
from skimage.morphology import erosion, skeletonize, thin
from skimage.filters import threshold_otsu
from google.colab import files
from PIL import Image
import io

# ✅ Fungsi Hit-or-Miss transform
def hit_or_miss(image, se_foreground, se_background):
    image_complement = invert(image)
    eroded_foreground = erosion(image, se_foreground)
    eroded_background = erosion(image_complement, se_background)
    hitmiss_result = eroded_foreground & eroded_background
    return hitmiss_result

# ✅ Upload file
print("Silakan upload gambar (JPG/PNG)...")
uploaded = files.upload()

# ✅ Ambil nama file
file_path = next(iter(uploaded))

# ✅ Baca dan konversi ke grayscale
img = Image.open(io.BytesIO(uploaded[file_path])).convert("L")
img_np = np.array(img)

# ✅ Binarisasi otomatis (Otsu)
thresh = threshold_otsu(img_np)
binary_image = img_np > thresh

# ✅ Structuring element otomatis
se_foreground = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 0]
], dtype=bool)

se_background = np.array([
    [1, 0, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=bool)

# ✅ Proses Hit-or-Miss
hitmiss_result = hit_or_miss(binary_image, se_foreground, se_background)

# ✅ Skeletonization
skeleton = skeletonize(binary_image)

# ✅ Thinning
thinning = thin(binary_image)

# ✅ Visualisasi semua hasil
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

axs[0, 0].imshow(binary_image, cmap='gray')
axs[0, 0].set_title("Binary Image")

axs[0, 1].imshow(hitmiss_result, cmap='gray')
axs[0, 1].set_title("Hit-or-Miss Result")

axs[0, 2].imshow(invert(binary_image), cmap='gray')
axs[0, 2].set_title("Complement Image")

axs[1, 0].imshow(skeleton, cmap='gray')
axs[1, 0].set_title("Skeletonization")

axs[1, 1].imshow(thinning, cmap='gray')
axs[1, 1].set_title("Thinning")

axs[1, 2].axis('off')  # Kosong

for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
```
Hasil program tersebut
![image](https://github.com/user-attachments/assets/465e0169-1b6d-41b7-9ad6-e50cbe69362b)
![image](https://github.com/user-attachments/assets/b73ba8b6-629a-4e89-a8d0-c99a33dc65cb)



