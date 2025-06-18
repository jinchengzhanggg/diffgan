import os
import time
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image,ImageOps
import torchvision
from torchvision.transforms import transforms
def fill_image():
    img_all_path = os.listdir(r"C:\Users\12908\Desktop\images\images")
    new_width = 256
    new_hight = 128
    for i in img_all_path:
        if not i.endswith("t"):
            image = Image.open(os.path.join(r"C:\Users\12908\Desktop\images\images",i))
            if image.mode == "P" or image.mode == "RGBA":
                image=image.convert("RGB")
            width = image.size[0]
            hight = image.size[1]
            if width >= hight:
                image_1 = image.resize((new_width,int(hight * new_width/width)))
                image_1_height = image_1.size[1]
                if image_1_height < new_hight:
                    differential_value = new_hight - image_1_height
                    if differential_value % 2 == 0 :
                        image_1 = ImageOps.expand(image_1,(0,differential_value//2,0,differential_value//2))
                    else:
                        image_1 = ImageOps.expand(image_1, (0, differential_value // 2, 0, differential_value // 2+1))
                elif image_1_height > new_hight:
                    left = 0
                    right = 256
                    bottom = (image_1_height+new_hight)/2
                    top = (image_1_height - new_hight )/2
                    image_1 = image_1.crop((left,top,right,bottom))
            else :
                image_1 = image.resize((int(width*new_hight/hight),new_hight))
                image_1_width = image_1.size[0]
                if image_1_width < new_width :
                    differential_value = new_width - image_1_width
                    if differential_value % 2 == 0 :
                        image_1 = ImageOps.expand(image_1,(differential_value//2,0,differential_value//2,0))
                    else:
                        image_1 = ImageOps.expand(image_1, (differential_value // 2, 0, differential_value // 2 +1 , 0))
                elif image_1_width > new_width :
                    top = 0
                    bottom = 128
                    right = (image_1_width + new_width) / 2
                    left = (image_1_width - new_width) / 2
                    image_1 = image_1.crop((left, top, right, bottom))

            image_1.save(f"../data/Pet/{i}")

def test_result():
    image_path = os.listdir("../data/Pet")
    for i in image_path:
        image = Image.open(os.path.join("../data/Pet",i))
        if image.size!=(256,128):
            print(image.size)
    print("无")

def generate_label():
    image_path = os.listdir("../data/Pet")
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

class Pet(Dataset):
    def __init__(self,root,train=True,transform=None,):
        self.path = root
        if not os.path.exists(root):
            print("路径不存在")
        self.class_path_label = os.listdir(root)
        self.class_num = []
        for i in self.class_path_label:
            if os.path.isdir(os.path.join(root,i)):
                self.class_num.append(i)
        self.label_index = {}
        num = 0
        self.img = []
        self.label = []
        for  per in self.class_num:
            per_class_path = os.path.join(root,per)
            for img_name in os.listdir(per_class_path):
                image = Image.open(os.path.join(per_class_path,img_name))
                self.img.append(image)
                self.label.append(num)
            self.label_index[per]=self.label_index.get(per,num)
            num+=1
        self.transform = transform

    def __getitem__(self, item):
        sample = self.img[item]
        label  = self.label[item]
        sample = self.transform(sample)
        return sample,label
    def __len__(self):
        return len(self.label)

if __name__ == '__main__':
    # fill_image()
    # test_result()
    generate_label()
    # train_transform = transforms.Compose([
    #     # transforms.Resize(args.image_size),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # pet = Pet("../data/Pet4",transform=train_transform)
    # dataloder = DataLoader(pet,batch_size=201)
    # for x,y in dataloder:
    #     torchvision.utils.save_image(x,"../test.jpg",normalize=True)
    #     print(y.shape)
    #     exit()
