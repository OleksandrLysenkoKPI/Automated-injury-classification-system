import numpy as np
import pydicom
import matplotlib.pyplot as plt

def plot_comparison(dicom_img, npy_img):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Оригінальний DICOM")
    plt.imshow(dicom_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Завантажений NumPy")
    plt.imshow(npy_img, cmap='gray')
    plt.axis('off')

    plt.show()

dicom_path = 'data/test_data/series-00000/image-00006.dcm'
ds = pydicom.dcmread(dicom_path)
dicom_pixels = ds.pixel_array

npy_path = './image-00006.npy'
npy_pixels = np.load(npy_path)

# Перевірка розмірності та типу даних
print(f"DICOM shape: {dicom_pixels.shape}, dtype: {dicom_pixels.dtype}")
print(f"NumPy shape: {npy_pixels.shape}, dtype: {npy_pixels.dtype}")

if dicom_pixels.shape == npy_pixels.shape:
    print("✅ Розміри збігаються")
else:
    print("❌ Розміри НЕ збігаються!")

# Перевірка значень (Min, Max, Mean)
print(f"DICOM range: [{dicom_pixels.min()}, {dicom_pixels.max()}], mean: {dicom_pixels.mean():.2f}")
print(f"NumPy range: [{npy_pixels.min()}, {npy_pixels.max()}], mean: {npy_pixels.mean():.2f}")

# Перевірка на повну ідентичність
if np.array_equal(dicom_pixels, npy_pixels):
    print("✅ Масиви ідентичні")
else:
    # При Rescale Slope/Intercept, вони можуть не бути ідентичними
    diff = np.abs(dicom_pixels.astype(float) - npy_pixels.astype(float))
    print(f"⚠️ Масиви мають розбіжності. Максимальна різниця: {np.max(diff)}")
    

plot_comparison(dicom_pixels, npy_pixels)