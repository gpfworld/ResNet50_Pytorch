from torch.utils import data
from torchvision import transforms as T
import os
from PIL import Image
from Config import config as cfg
import torch as t
import random
from utils.tool import DrawImage

class TrademarkImageDataSet(data.Dataset):
	def __init__(self,root_img ,root_label = None ,transforms = None,train = True,test=False):
		
		self.train = train
		self.test = test
		
		if root_label:
			with open(root_label) as f:
				imgs = [img.strip('\n').split(' ') for img in f]
			
		self.root_img = root_img

		self.imgs = [ img[0] for img in imgs]
		self.labels = [ int(img[1]) for img in imgs]
		
		
		if transforms is None:
			normalize = T.Normalize(mean = [0.485,0.456,0.406] , std = [0.229, 0.224, 0.225])
			
			if self.test or not train:
				self.transforms = T.Compose([
					T.Resize((260, 550)),
					T.FiveCrop((200, 400)),
					(lambda crops: t.stack([normalize(T.ToTensor()(crop)) for crop in crops]))
				])
			else:
				self.transforms = T.Compose(
					[T.Resize((260,550)),
					 T.RandomCrop((200,400)),
					 T.ToTensor(),
					 normalize]
				)
		else:
			self.transforms = transforms
		
	def __getitem__(self, index):
		
		img_path = os.path.join(self.root_img,self.imgs[index])
		label = int(self.labels[index])
		
		data = Image.open(img_path)
		while data is None:
			img_path = self.imgs[t.round(t.rand(1)* len(self.imgs))]
			data = Image.open(img_path)
		if self.test or not self.train:
			label = t.FloatTensor([label]).expand(5).unsqueeze(1)
		data = self.transforms(data)
		return data,label
	
	def __len__(self):
		return len(self.imgs)
		
	def get_item(self,index):
		img_path = os.path.join(self.root_img, self.imgs[index])
		label = int(self.labels[index])
		
		return img_path, label
	
if __name__=="__main__":
	cfg  = cfg()
	# dataset = TrademarkImageDataSet(cfg.train_img_dir,cfg.train_val_label_path)
	# for data,label in dataset:
	# 	print(label)

	val_data = TrademarkImageDataSet(cfg.test_img_dir, cfg.test_label_path,test=True )

	val_dataloader = data.DataLoader(val_data, batch_size= cfg.test_batchsize , shuffle=False, num_workers=cfg.num_workers)
	
	# for data,label in val_data:
	# 	print (label )
	# data,label = val_data[4]
	#
	# print(data)
	# print(label)
	
	# data = T.Resize((260,550))(data)


	# print(len(val_data))
	# for ii in range(len(val_data)):
	# 	print(val_data.get_item(ii))
	
	data, label = next(iter(val_dataloader))
	print(label.size())
	print(data.size())