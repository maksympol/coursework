import numpy as np
from glob import glob
from tqdm import tqdm
import nrrd
from PIL import Image
import os

output_size = [112, 112, 80]

def convert_to_jpg():
    listt = glob('../../LA_dataset/cardiography/*/lgemri.nrrd')
    for item in tqdm(listt):
        image, img_header = nrrd.read(item)
        label, gt_header = nrrd.read(item.replace('lgemri.nrrd', 'laendo.nrrd'))
        label = (label == 255).astype(np.uint8)
        w, h, d = label.shape

        tempL = np.nonzero(label)
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        image = image[minx:maxx, miny:maxy, minz:maxz]
        label = label[minx:maxx, miny:maxy, minz:maxz]

        # Створюємо вихідну папку, якщо її не існує
        output_dir = item.replace('lgemri.nrrd', '')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Зберігаємо зображення та мітки як файли .jpg по шарам (з третьої осі)
        for i in range(image.shape[2]):
            img_slice = Image.fromarray((image[:, :, i] * 255).astype(np.uint8))
            label_slice = Image.fromarray((label[:, :, i] * 255).astype(np.uint8))
            img_slice.save(os.path.join(output_dir, f'image_{i:03d}.jpg'))
            label_slice.save(os.path.join(output_dir, f'label_{i:03d}.jpg'))

if __name__ == '__main__':
    convert_to_jpg()
