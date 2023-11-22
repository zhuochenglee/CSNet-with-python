import os
import random
from tqdm import tqdm
from PIL import Image


def preprocess_image(image_path):
    # 读取原始图像
    cropped_images = []
    for i in tqdm(image_path, desc="读取图片"):
        image = Image.open(i)
        # 裁剪图像为96x96大小
        width, height = image.size
        stride = 57
        size = 96

        for top in range(0, height - size + 1, stride):
            for left in range(0, width - size + 1, stride):
                right = left + size
                bottom = top + size
                cropped_image = image.crop((left, top, right, bottom))
                cropped_images.append(cropped_image)
    return cropped_images


def flip_image(image):
    # 水平翻转图像
    flipped_image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    return flipped_image


def rotate_image(image, angle):
    # 旋转图像
    rotated_image = image.rotate(angle)
    return rotated_image


def process_images(image_path):
    # 处理图像
    images = preprocess_image(image_path)
    processed_images = []

    for image in tqdm(images, desc="处理图片"):
        processed_images.append(image)
        processed_images.append(flip_image(image))
        processed_images.append(rotate_image(image, 90))
        processed_images.append(rotate_image(flip_image(image), 90))
        processed_images.append(rotate_image(image, 180))
        processed_images.append(rotate_image(flip_image(image), 180))
        processed_images.append(rotate_image(image, 270))
        processed_images.append(rotate_image(flip_image(image), 270))

    return processed_images


def get_image_files(folder_path):
    image_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png']
    for file_name in os.listdir(folder_path):
        file_extension = os.path.splitext(file_name)[1].lower()
        if file_extension in valid_extensions:
            image_files.append(os.path.join(folder_path, file_name))
    return image_files

def random_select(images, num_images):
    random_images = random.sample(images, num_images)
    return random_images



save_dir = 'BSDS500/processed_images'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

folder_path = "BSDS500/train"
image_files = get_image_files(folder_path)
processed_images = process_images(image_files)
print(f'total images: {len(processed_images)}')
select_images = random_select(processed_images, 5000)

for i, image in tqdm(enumerate(select_images), total=len(select_images), desc="保存图片"):
    image.save(os.path.join(save_dir, str(i) + '.png'))

print(f"保存完成，等待程序结束")