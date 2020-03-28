# TODO: 



import os
import random 
import time
import numpy
import torch
import math
import pdb

import sys
import numpy as np
from PIL import Image, ImageOps

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage, Resize

from dataloader import SemanticSegmentation
from config import *
from SqueezeSeg import SqueezeSeg
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

	def forward(self, outputs, targets):
		return self.loss(torch.nn.functional.log_softmax(outputs, dim=1),targets)

def train(model, enc=False):
	best_acc = 0

	print('Total Number of classes is {}'.format(NUM_CLASSES))

	co_transforms = ImageTransform(width=IMG_WIDTH)

	iouEvalTrain = iouEval(NUM_CLASSES)

	dataset_train = SemanticSegmentation( 
					root = ROOT_DIR,
					split = 'train',
					co_transforms = co_transforms)

	dataset_val = SemanticSegmentation(
					root = ROOT_DIR,
					split = 'test',
					co_transforms = co_transforms)

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

	weight, num_images = load_class_weights()
	print('Imbalance weights',weight)
	if ARGS_CUDA:
		weight = weight.cuda()
		
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

	for epoch in range(ARGS_NUM_EPOCHS+1):
		print("\n ---------------- [TRAINING] Epoch #", epoch, "------------------\n")
		epoch_loss = []
		scheduler.step(epoch)
		model.train()
		iouEvalTrain = iouEval(NUM_CLASSES)
		iouEvalval = iouEval(NUM_CLASSES)
		for step, (image,label) in enumerate(loader):
			
			start_time = time.time()

			if ARGS_CUDA:
				image = image.cuda()
				label = label.cuda()

			image = Variable(image)
			label = Variable(label)

			output = model(image)


			iouEvalTrain.addBatch(
				output.max(1)[1].unsqueeze(1).data,
				label.data
			)

			optimizer.zero_grad()
			loss = criterion(output,label[:,0])

			loss.backward()
			optimizer.step()

			epoch_loss.append(loss.item())
			
		
		avg_loss = sum(epoch_loss) / len(epoch_loss)
		iouTrain, iou_classes = iouEvalTrain.getIoU()
		
		print('[Average loss]:{avgloss} [Average IOU]:{iou_train} [bg]:{bg} [Car]:{car} [Ped]:{ped} [Bicy]:{bicy}'.format(
					avgloss=avg_loss,
					iou_train = iouTrain,
					bg = iou_classes[0],
					car = iou_classes[1],
					ped = iou_classes[2],
					bicy = iou_classes[3]))

		model.eval()


		print("\n ---------------- [VALIDATING] Epoch #", epoch, "------------------\n")

		for step, (image,label) in enumerate(loader_val):
			start_time = time.time()

			if ARGS_CUDA:
				image = image.cuda()
				label = label.cuda()

			image = Variable(image)
			label = Variable(label)

			output = model(image)


			iouEvalval.addBatch(
				output.max(1)[1].unsqueeze(1).data,
				label.data
			)

		iouVal, iou_classes = iouEvalval.getIoU()
		print('[bg]:{bg} [Car]:{car} [Ped]:{ped} [Bicy]:{bicy}'.format(
					bg = iou_classes[0],
					car = iou_classes[1],
					ped = iou_classes[2],
					bicy = iou_classes[3]))

if __name__ == '__main__':
	model = SqueezeSeg(NUM_CLASSES)
	if ARGS_CUDA:
		model = model.cuda()

	train(model)
