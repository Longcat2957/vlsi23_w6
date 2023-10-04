import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

# 이미지를 읽는 함수를 정의
def get_image(p:str):
    return Image.open(p).convert("RGB")

class baseDataset(Dataset):
    def __init__(self, root:str="./MNIST", istrain:bool=True):
        super().__init__()
        root = os.path.join(root, 'train') if istrain else os.path.join(root, 'val')
        
        # 데이터 리스트 생성
        data_list = []
        for i in range(10):
            class_index = str(i)
            class_dir = os.path.join(root, class_index)
            for img in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img)
                data_list.append((i, img_path))

        self.data_list = data_list
        
        # 훈련과 검증과정에서 서로 다른 augmentatation 기법을 사용
        if istrain:
            self.transform = T.Compose([
                T.RandomCrop(28),  # 이미지를 랜덤하게 자르고 크기를 32x32로 조정
                T.RandomRotation(15),  # 이미지를 랜덤하게 회전
                T.ToTensor(),  # 이미지를 텐서로 변환
            ])
        else:
            self.transform = T.Compose([
                T.CenterCrop(28),
                T.ToTensor(),
            ])
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx:int):
        # 인덱스에 해당하는 데이터 반환
        number, img_path = self.data_list[idx]
        img_obj = get_image(img_path)
        img_tensor = self.transform(img_obj)
        
        return img_tensor, number

def get_dataloader(root:str, batch_size:int=32, num_workers:int=2):
    train_dataset, val_dataset = baseDataset(root, True), baseDataset(root, False)
    train_loader, val_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers), \
        DataLoader(val_dataset, batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, val_loader

if __name__ == "__main__":
    # test baseDataset Class
    train_dataset, val_dataset = baseDataset("./MNIST", True), baseDataset("./MNIST", False)
    print(f"# Length of data, TRAIN : {len(train_dataset)}, VAL : {len(val_dataset)}")
    
    # test tensor shape
    tensor, number = train_dataset[200]
    print(f"# Tensor Shape(unbatched) : {tensor.shape}")