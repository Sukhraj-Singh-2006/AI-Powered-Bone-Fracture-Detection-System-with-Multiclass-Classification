from PIL import Image
import os

dataset_path = "dataset"

bad_images = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(("png","jpg","jpeg")):
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()
            except:
                print("Deleting corrupted:", path)
                bad_images.append(path)
                os.remove(path)

print("Done cleaning dataset.")