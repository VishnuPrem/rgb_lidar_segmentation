# TODO: tensorboardX
# TODO: Best model saver
# TODO: use pretrained backbone from pytorch



import os
import random 
import time
import numpy
import torch
import math
import pdb

import sys
import numpy as np
import torch.nn as nn
from PIL import Image, ImageOps
import torchvision.models as models

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage, Resize

from dataloader import Squeeze_Seg
from config import *

sys.path.append(ARGS_ROOT +'/models/'+ ARGS_MODEL_NAME)
from SqueezeSeg import SqueezeSeg 
#from models.SqueezeSeg.SqueezeSeg import SqueezeSeg
from utils.util_iou_eval import iouEval, getColorEntry
from utils.calculate_weights import load_class_weights

class ImageTransform(object):
	def __init__(self,width):
		self.width = width

	def __call__(self, input,target):
		input = Resize((64,self.width), Image.BILINEAR)(input)
		target = Resize((64,self.width),Image.NEAREST)(target)

		sup_labels = np.array(target,dtype=np.int8)-7
		[h,w] = sup_labels.shape

		sup_labels[np.logical_or(sup_labels<=0,sup_labels>=4)]=0
		target = Image.fromarray(sup_labels)
		
		#th = 64
		#tw = 512

		#x1 = random.randint(0, w - tw)
		#y1 = random.randint(0, h - th)

		#target = target.crop((x1, y1, x1 + tw, y1 + th))
		#input = input.crop((x1, y1, x1 + tw, y1 + th))
		#print(input.size,target.size)
		input = ToTensor()(input)
		target = torch.from_numpy(np.array(target)).long().unsqueeze(0)

		return input, target

class CrossEntropyLoss2d(torch.nn.Module):
	def __init__(self, weight=None):
		super().__init__()
		self.loss = torch.nn.NLLLoss(weight=weight)

	def forward(self, outputs, targets,mask=None):
		return self.loss(torch.nn.functional.log_softmax(outputs, dim=1),targets)

class brc_loss(torch.nn.Module):
	def __init__(self, weight=None):
		super().__init__()
		self.loss = torch.nn.NLLLoss(weight=weight)
	
	def forward(self, output,target,mask):
		#pdb.set_trace()
		loss = self.loss(torch.nn.functional.log_softmax(output, dim=1),target)
		loss = mask.view(-1,)*loss
		loss = torch.sum(loss)/torch.sum(mask)

		return loss*CLS_LOSS_COEF



def load_pretrained(model,squeezenet):
	name_squeezenet = [i for i,_ in squeezenet.state_dict().items()]
	param_squeezenet = [i for _,i in squeezenet.state_dict().items()]
	names_model = [i for i,_ in model.state_dict().items()]
	
	name_squeezenet.insert(2,0)
	name_squeezenet.insert(2,0)
	param_squeezenet.insert(2,0)
	param_squeezenet.insert(2,0)
	
	if len(names_model)>100:
		name_squeezenet = name_squeezenet[0:52]*2
		param_squeezenet = param_squeezenet[0:52]*2

	new_list = [0]*50
	name_squeezenet.extend(new_list)


	param_squeezenet.extend(new_list)
	i = 0
	
	new_dict = model.state_dict().copy()
	for name, param in model.state_dict().items():
		squeeze_name = name_squeezenet[i]
		squeeze_param = param_squeezenet[i]
		if squeeze_name == 0:
			pass
		elif name == 'fire10.layer_1.conv.weight':
			pass
		elif name == 'fire10.layer_1.conv.bias':
			pass
		else:
			#print(name)
			#j+=1
			new_dict[name] = squeeze_param
		i += 1
	
	model.load_state_dict(new_dict)


def train(model, enc=False):
	best_acc = 0

	print('Total Number of classes is {}'.format(NUM_CLASSES))

	co_transforms = ImageTransform(width=IMG_WIDTH)

	iouEvalTrain = iouEval(NUM_CLASSES)

	dataset_train = Squeeze_Seg(ROOT_DIR,'train',ARGS_INPUT_TYPE_1,ARGS_INPUT_TYPE_2)

	dataset_val = Squeeze_Seg(ROOT_DIR,'val',ARGS_INPUT_TYPE_1,ARGS_INPUT_TYPE_2)

	loader = DataLoader(
		dataset_train,
		num_workers = ARGS_NUM_WORKERS,
		batch_size = ARGS_TRAIN_BATCH_SIZE,
		shuffle = True)

	loader_val = DataLoader(
		dataset_val,
		num_workers = ARGS_NUM_WORKERS,
		batch_size = ARGS_VAL_BATCH_SIZE,
		shuffle = True)

	weight = load_class_weights()
	print('Imbalance weights',weight)
	if ARGS_CUDA:
		weight = weight.cuda()
	if ARGS_BRC:	
		criterion = brc_loss(weight=weight)
	else:
		criterion = CrossEntropyLoss2d(weight=weight)
	savedir = ARGS_SAVE_DIR + ARGS_MODEL

	if not os.path.exists(ARGS_SAVE_DIR):
		os.mkdir(ARGS_SAVE_DIR)

	if not os.path.exists(savedir):
		os.mkdir(savedir)

	with open(savedir + "model.txt","w") as myfile:
		myfile.write(str(model))

	optimizer = Adam(
		model.parameters(),
		OPT_LEARNING_RATE_INIT,
		OPT_BETAS,
		eps = OPT_EPS_LOW,
		weight_decay = OPT_WEIGHT_DECAY 
		)

	lambda1 = lambda epoch: pow((1-((epoch-1)/ARGS_NUM_EPOCHS)),0.9)
	scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

	best_iou = 0

	for epoch in range(ARGS_NUM_EPOCHS+1):
		print("\n ---------------- Epoch #", epoch, "------------------\n")
		epoch_loss = []
		val_epoch_loss = []
		
		model.train()
		iouEvalTrain = iouEval(NUM_CLASSES)
		iouEvalval = iouEval(NUM_CLASSES)
		
		for step, (image,image_2,mask,label) in enumerate(loader):
			
			start_time = time.time()

			if ARGS_CUDA:
				image = image.cuda()
				image_2 = image_2.cuda()
				label = label.cuda()
				mask = mask.cuda()

			image = Variable(image)
			label = Variable(label)
			mask = Variable(mask)
			image_2 = Variable(image_2)

			output = model(image,image_2,mask)


			iouEvalTrain.addBatch(
				output.max(1)[1].unsqueeze(1).data,
				label.data
			)

			optimizer.zero_grad()
			loss = criterion(output,label[:,0],mask)

			loss.backward()
			optimizer.step()
			scheduler.step(epoch)
			epoch_loss.append(loss.item())
			
		
		avg_loss = sum(epoch_loss) / len(epoch_loss)
		iouTrain, iou_classes = iouEvalTrain.getIoU()
		
		print('[TRAINING] [Average loss]:{avgloss} [Average IOU]:{iou_train} [bg]:{bg} [Car]:{car} [Ped]:{ped} [Bicy]:{bicy}'.format(
					avgloss=avg_loss,
					iou_train = iouTrain,
					bg = iou_classes[0],
					car = iou_classes[1],
					ped = iou_classes[2],
					bicy = iou_classes[3]))

		model.eval()


		#print("\n ----------------  Epoch #", epoch, "------------------\n")

		for step, (image,mask,label) in enumerate(loader_val):
			start_time = time.time()

			if ARGS_CUDA:
				image = image.cuda()
				label = label.cuda()
				mask = mask.cuda()

			image = Variable(image)
			label = Variable(label)
			mask = Variable(mask)

			output = model(image,mask)
			loss = criterion(output,label[:,0],mask)

			val_epoch_loss.append(loss.item())

			iouEvalval.addBatch(
				output.max(1)[1].unsqueeze(1).data,
				label.data
			)

		iouVal, iou_classes = iouEvalval.getIoU()
		avg_loss = sum(val_epoch_loss) / len(val_epoch_loss)
		
		print('[VALIDATING] [loss]:{loss} [avg_iou]:{iou} [bg]:{bg} [Car]:{car} [Ped]:{ped} [Bicy]:{bicy}'.format(
					loss = avg_loss,
					iou =  iouVal,
					bg = iou_classes[0],
					car = iou_classes[1],
					ped = iou_classes[2],
					bicy = iou_classes[3]))

		if iouVal>best_iou:
			torch.save(model.state_dict(), savedir + '/model_best.pth')
			print('[SAVED] Best Model')
			best_iou = iouVal

if __name__ == '__main__':
	model = SqueezeSeg(data_dict)
	
	if ARGS_PRETRAINED:
		squeezenet = models.squeezenet1_1(pretrained=True)
		load_pretrained(model,squeezenet)
	if ARGS_MODEL_NAME=='Dual_SqueezeSeg/':
		model.conv1_1.conv = nn.Conv2d(len(ARGS_INPUT_TYPE_1), 64, 3, stride=(1,2), padding=1)
		model.conv1_1_skip.conv = nn.Conv2d(len(ARGS_INPUT_TYPE_1),64, 1, stride=1, padding=0)

		model.conv2_1.conv = nn.Conv2d(len(ARGS_INPUT_TYPE_2), 64, 3, stride=(1,2), padding=1)
		model.conv2_1_skip.conv = nn.Conv2d(len(ARGS_INPUT_TYPE_2),64, 1, stride=1, padding=0)
	else:
		model.conv1.conv = nn.Conv2d(len(ARGS_INPUT_TYPE_1), 64, 3, stride=(1,2), padding=1)
		model.conv1_skip.conv = nn.Conv2d(len(ARGS_INPUT_TYPE_1),64, 1, stride=1, padding=0)
	torch.set_num_threads(ARGS_NUM_WORKERS)


	if ARGS_CUDA:
		model = model.cuda()

	train(model)
