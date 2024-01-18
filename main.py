import os
import torch
from torchvision import models, transforms
from PIL import Image

images_folder = '/Users/jessica.kim/Desktop/macmillanCovers'
images = os.listdir(images_folder)

weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)
model.eval()\

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

for i in range(len(images)):
    current_image = images[i]

    if current_image.lower() == '.ds_store':
        continue

    try:
        image = Image.open(os.path.join(images_folder, images[i]))
        input_tensor = transform(image).unsqueeze(0)

        if input_tensor.size()[1] == 3:
            output_tensor = model(input_tensor)
            predicted_category = weights.meta["categories"][torch.argmax(output_tensor)]
            print(f'{current_image} predicted as {predicted_category}')

    except Exception as e:
        print(f'Skipping {current_image} due to error: {e}')
