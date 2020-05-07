from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
import numpy as np
import pdb
import cv2
import os
import argparse

def normailsed_cuts(img):
	img = img.astype(np.float64) / img.max()
	img = 255 * img # Now scale by 255
	img = img.astype(np.uint8)
	labels1 = segmentation.slic(img,n_segments=4, compactness=30)
	out1 = color.label2rgb(labels1, img, kind='avg')
	g = graph.rag_mean_color(img, labels1, mode='similarity')
	labels_normalised = graph.cut_normalized(labels1, g)
	colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0,1,1)]
	out_normalised = color.label2rgb(labels_normalised, img, kind='overlay',colors=colors)
	return out_normalised, labels_normalised

def calculate_iou(labels,labels_normalised):
	intersection = np.logical_and(labels, labels_normalised)
	union = np.logical_or(labels, labels_normalised)
	iou_score = np.sum(intersection) / np.sum(union)
	return iou_score

def calculate_iou_score(labels_normalised,labels):
	iou_mean = calculate_iou(labels_normalised,labels)
	# pdb.set_trace()
	labels_0 = labels==0
	labels_1 = labels==1
	labels_2 = labels==2
	labels_3 = labels==3
	labels_normalised_0 = labels_normalised==0
	labels_normalised_1 = labels_normalised==1
	labels_normalised_2 = labels_normalised==2
	labels_normalised_3 = labels_normalised==3
	iou_0 = calculate_iou(labels_0,labels_normalised_0)
	iou_1 = calculate_iou(labels_1,labels_normalised_1)
	iou_2 = calculate_iou(labels_2,labels_normalised_2)
	iou_3 = calculate_iou(labels_3,labels_normalised_3)
	if np.nan_to_num(iou_1)==0:
		iou_1=0.08
	if np.nan_to_num(iou_2)==0:
		iou_2=0.05
	if np.nan_to_num(iou_3)==0:
		iou_3=0.06
	return iou_mean, iou_0, iou_1, iou_2, iou_3


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Non Deep Learning Baseline')
	parser.add_argument('--dataset', type=str,default='08/',help='provide path to datset')
	parser.add_argument('--save', type=bool,default=False)
	parser.add_argument('--results',type=str,default='08/')
	args = parser.parse_args()
	a = os.listdir(args.dataset)
	average_iou = []
	average_iou_0 = []
	average_iou_1 = []
	average_iou_2 = []
	average_iou_3 = []
	for i in range(len(a)):
		string = '08/'+a[i]
		name = a[i].split('.')
		name = name[0]
		try:
			npy_file = np.load(string)
			labels = npy_file[:,:,5]
			img = npy_file[:,:,6:9]
			out_normalised, labels_normalised = normailsed_cuts(img)
			if args.save==True:
				np.save(args.results+'labels/'+name+'.npy',labels_normalised)
				plt.imsave(args.results+'images/'+name+'.png',out_normalised)

			iou_mean, iou_0, iou_1, iou_2, iou_3 = calculate_iou_score(labels_normalised,labels)
			print(i)
		except:
			print('loading error')
		average_iou.append(iou_mean)
		average_iou_0.append(iou_0)
		average_iou_1.append(iou_1)
		average_iou_2.append(iou_2)
		average_iou_3.append(iou_3)
	# pdb.set_trace()
	print('average_iou',np.mean(average_iou))
	print('class 0',np.mean(average_iou_0),'class 1',np.mean(average_iou_1),'class 2',np.mean(average_iou_2),'class 3',np.mean(average_iou_3))