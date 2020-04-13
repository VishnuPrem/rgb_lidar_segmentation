import os
import pdb
import torch
import random
import numpy as np
from os import walk
from config import *
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage, Resize, RandomCrop
from utils.calculate_weights import load_datastats


def load_image(file):
	return Image.open(file)

def image_path(root, basename):
	return os.path.join(root,basename)

class Image_SemanticSegmentation:
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

class Squeeze_Seg:
	def __init__(self, root,split,format_,co_transforms=None):
		self.root = os.path.join(root, split)

		self.data_stats = load_datastats()
		print('[STATS] \n[Mean]:    {} \n[Std_dev]: {}'.format(self.data_stats[0,:,0].T,self.data_stats[0,:,1].T))

		self.format = format_

		self.image_list=[]
		for (dirpath, dirnames, filenames) in walk(self.root):
			for file in filenames:
				if file[-4:]=='.npy':
					self.image_list+=[os.path.join(dirpath,file)]
		
		self.image_list.sort()
		self.co_transform = co_transforms
		print('[Dataloader] Loaded {} files'.format(len(self.image_list)))
		
	def __getitem__(self,index):
		filename = self.image_list[index]

		data = np.load(filename) # x,y,z,reflectance,depth,label,rgb
		data_rep = {}
		
		invalid_points = np.argwhere(data[:,:,0]==-1)
		data_rep['X'] = ToTensor()(data[:,:,0]-self.data_stats[0,0,0])/self.data_stats[0,0,1]
		data_rep['Y'] = ToTensor()(data[:,:,1]-self.data_stats[0,1,0])/self.data_stats[0,1,1]
		data_rep['Z'] = ToTensor()(data[:,:,2]-self.data_stats[0,2,0])/self.data_stats[0,2,1]
		data_rep['I'] = ToTensor()(data[:,:,3]-self.data_stats[0,3,0])/self.data_stats[0,3,1]
		data_rep['D'] = ToTensor()(data[:,:,4]-self.data_stats[0,4,0])/self.data_stats[0,4,1]
		data_rep['R'] = ToTensor()(data[:,:,6])
		data_rep['G'] = ToTensor()(data[:,:,7])
		data_rep['B'] = ToTensor()(data[:,:,8])
		

		data_rep['X'][0][invalid_points.T], \
		data_rep['Y'][0][invalid_points.T], \
		data_rep['Z'][0][invalid_points.T], \
		data_rep['I'][0][invalid_points.T], \
		data_rep['D'][0][invalid_points.T] = -1,-1,-1,-1,-1

		
		lidar_mask = torch.from_numpy((data[:,:,4]>0)*1.).float().unsqueeze(0)

		inputs = torch.zeros((len(self.format),64,512),dtype = torch.float)
		
		for val,i in enumerate(self.format):
			inputs[val]=data_rep[i]

		data[:,:,5][data[:,:,5]==-1]=0
		label = torch.from_numpy(data[:,:,5]).long().unsqueeze(0)
			

		return inputs,lidar_mask,label

	def __len__(self):
		return len(self.image_list)


if __name__ == "__main__":

	dataset_train = Squeeze_Seg('/home/neil/cis_522/squeezeSeg/data/','train',ARGS_INPUT_TYPE)
	
	loader = DataLoader(
		dataset_train,
		num_workers = 6,
		batch_size = 10,
		shuffle = True)
	

	for image,mask,label in loader:
		print(image.shape,label.shape,mask.shape)
		pdb.set_trace()
