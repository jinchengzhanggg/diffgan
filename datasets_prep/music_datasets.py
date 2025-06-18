import os
import time
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image,ImageOps,ImageFile
import torchvision
from torchvision.transforms import transforms
import numpy as np


ImageFile.LOAD_TRUNCATED_IMAGES = True
def fill_image(root,new_width=None,new_height=None):
    img_all_paths = os.listdir(root)
    if new_height is None:
        new_height = 128
    if new_width is None:
        new_width = 256
    for img_all_path in img_all_paths:
        class_img = os.listdir(os.path.join(root,img_all_path))
        for i in class_img:
            if not i.endswith("t"):
                image = Image.open(os.path.join(root,img_all_path,i))
                if image.mode == "P" or image.mode == "RGBA":
                    image=image.convert("RGB")
                width = image.size[0]
                hight = image.size[1]
                if width >= hight:
                    image_1 = image.resize((new_width,int(hight * new_width/width)))
                    image_1_height = image_1.size[1]
                    if image_1_height < new_height:
                        differential_value = new_height - image_1_height
                        if differential_value % 2 == 0 :
                            image_1 = ImageOps.expand(image_1,(0,differential_value//2,0,differential_value//2))
                        else:
                            image_1 = ImageOps.expand(image_1, (0, differential_value // 2, 0, differential_value // 2+1))
                    elif image_1_height > new_height:
                        left = 0
                        right = new_width
                        bottom = (image_1_height + new_height) / 2
                        top = (image_1_height - new_height) / 2
                        image_1 = image_1.crop((left, top, right, bottom))
                else :
                    image_1 = image.resize((int(width*new_height/hight),new_height))
                    image_1_width = image_1.size[0]
                    if image_1_width < new_width :
                        differential_value = new_width - image_1_width
                        if differential_value % 2 == 0 :
                            image_1 = ImageOps.expand(image_1,(differential_value//2,0,differential_value//2,0))
                        else:
                            image_1 = ImageOps.expand(image_1, (differential_value // 2, 0, differential_value // 2 +1 , 0))

                    elif image_1_width > new_width:
                        top = 0
                        bottom = new_height
                        right = (image_1_width + new_width) / 2
                        left = (image_1_width - new_width) / 2
                        image_1 = image_1.crop((left, top, right, bottom))

                image_1.save(os.path.join(root,img_all_path,i))

def test_result(root,new_width,new_height):
    image_path = os.listdir(root)
    for i in image_path:
        image = Image.open(os.path.join(root,i))
        if image.size!=(new_width,new_height):
            print(image.size)
    print("无")

def generate_label_4(root):
    image_path = os.listdir(root)
    classified = ["Abyssinian","Bengal","american_bulldog","yorkshire"]
    for i in classified:
        if not os.path.exists(f"../data/Pet4/{i}"):
            os.mkdir(f"../data/Pet4/{i}")
    for i in image_path:
        if "Abyssinian" in i :
            image = Image.open(os.path.join("../data/Pet",i))

            image.save(f"../data/Pet4/Abyssinian/{i}")
        elif "Bengal" in i :
            image = Image.open(os.path.join("../data/Pet", i))
            image.save(f"../data/Pet4/Bengal/{i}")
        elif "american_bulldog" in i :
            image = Image.open(os.path.join("../data/Pet", i))
            image.save(f"../data/Pet4/american_bulldog/{i}")
        elif "yorkshire" in i :
            image = Image.open(os.path.join("../data/Pet", i))
            image.save(f"../data/Pet4/yorkshire/{i}")

def split_data(root,rate=0.1):
    img_path_all = os.listdir(root)
    two_datasets = ["train","test"]
    for i in two_datasets:
        if not os.path.exists(os.path.join(root,i)):
            os.mkdir(os.path.join(root,i))
            for j in img_path_all:
                os.mkdir(os.path.join(root,i,j))
    for per_class_name in img_path_all:
        per_class = os.listdir(os.path.join(root,per_class_name))
        for img_path in per_class:
            random_choice = torch.rand(1)>rate
            img = Image.open(os.path.join(root,per_class_name,img_path))
            if random_choice:
                img.save(os.path.join(root,two_datasets[0],per_class_name,img_path))
            else:
                img.save(os.path.join(root, two_datasets[1], per_class_name, img_path))
                



class Music(Dataset):
    def __init__(self,root):
        self.path = root
        if not os.path.exists(root):
            print("路径不存在")
        self.class_path_label = os.listdir(root)
        self.class_num = []
        for i in self.class_path_label:
            if os.path.isdir(os.path.join(root,i)):
                self.class_num.append(i)
        self.label_index = {}
        self.index_label = {}
        num = 0
        self.img = []
        self.label = []
        for  per in self.class_num:
            per_class_path = os.path.join(root,per)
            for img_name in os.listdir(per_class_path):
                image = np.load(os.path.join(per_class_path,img_name))[np.newaxis]
                self.img.append(image)
                self.label.append(num)
            self.label_index[per]=self.label_index.get(per,num)
            self.index_label[num]=self.index_label.get(num,per)
            num+=1
        
    def __len__(self):
        return len(self.img)
    def __getitem__(self, item):
        sample = self.img[item]
        label  = self.label[item]
        
        sample = sample.astype(np.float32) / 63.5 - 1
        sample = torch.from_numpy(sample)
        # sample[:, :, :21, :] = -1
        # sample[:, :, 108 + 1:, :] = -1
        sample[:, :, :21, :] = 0
        sample[:, :, 108 + 1:, :] = 0
        sample = sample.squeeze(0)
        real_data = sample.numpy()

    

        
 
        
        
        # sample = torch.cat((sample,torch.zeros(1,sample.shape[1],sample.shape[2])),dim=0)
        # sample = torch.cat((sample,-torch.ones(1,sample.shape[1],sample.shape[2])),dim=0)
        
        return sample,label


if __name__ == '__main__':
    fill_image("../data/Pet4/test",128,128)
    # split_data("../data/Pet4")
    pass
    # train_transform = transforms.Compose([
    #     # transforms.Resize(args.image_size),
    #     # transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # pet = Pet("../data/Pet4/train",transform=train_transform)
    # dataloder = DataLoader(pet,batch_size=1)
    # for x,y in dataloder:
    #     print(x.shape)
    #     print(y.shape)
    #     exit()