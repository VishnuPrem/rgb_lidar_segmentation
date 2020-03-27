import numpy as np
import os
import pdb
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor, ToPILImage, Resize, RandomCrop
from torch.utils.data import DataLoader

import random

def load_image(file):
	return Image.open(file)

def image_path(root, basename):
	return os.path.join(root,basename)

class SemanticSegmentation:
	def __init__(self, root,split,co_transforms=None):
		self.root = root
		with open(os.path.join(root,split+'.txt')) as f:
			self.image_list = [fi.strip().split(' ') for fi in f.readlines()]
		
		self.co_transform = co_transforms

	def __getitem__(self,index):
		filename = self.image_list[index]

		with open(image_path(self.root,filename[0]),'rb') as f:
			image = load_image(f).convert('RGB')
		with open(image_path(self.root,filename[1]),'rb') as f:
			label = load_image(f).convert('P')



		if self.co_transform:
			image,label = self.co_transform(image,label)

		return image,label

	def __len__(self):
		return len(self.image_list)




if __name__ == "__main__":
	co_transforms = ImageTransform(width=512)
	dataset_train = SemanticSegmentation('/home/neil/squeezeSeg/Camvid/CamVid/','train',co_transforms)
	
	loader = DataLoader(
		dataset_train,
		num_workers = 6,
		batch_size = 3,
		shuffle = True)
	

	for image ,label in loader:
		pdb.set_trace()