import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def prepare_dataset(data_path='data/caltech-101', train_size=0.7):
    for split in ['train', 'test']:
        os.makedirs(os.path.join(data_path, split), exist_ok=True)
    
    source_dir = os.path.join(data_path, '101_ObjectCategories')
    
    blacklist = {'BACKGROUND_Google'}  # 过滤非有效类别
    
    for class_dir in os.listdir(source_dir):
        if class_dir.startswith('.') or class_dir in blacklist:
            continue
            
        class_path = os.path.join(source_dir, class_dir)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
            train_imgs, test_imgs = train_test_split(images, train_size=train_size, random_state=2025)
            
            for img in tqdm(train_imgs, desc=f'{class_dir} train', total=len(train_imgs)):
                src = os.path.join(class_path, img)
                dst = os.path.join(data_path, 'train', class_dir)
                os.makedirs(dst, exist_ok=True)
                shutil.copy(src, dst)
            

            for img in tqdm(test_imgs, desc=f'{class_dir} test', total=len(test_imgs)):
                src = os.path.join(class_path, img)
                dst = os.path.join(data_path, 'test', class_dir)
                os.makedirs(dst, exist_ok=True)
                shutil.copy(src, dst)

if __name__ == '__main__':
    prepare_dataset()