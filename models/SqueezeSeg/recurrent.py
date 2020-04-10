import numpy as np	
import torch
import torch.nn as nn
import torch.nn.functional as F	
import utils.util_recurrent as util
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Recurrent(nn.Module):
	"""docstring for Recurrent
	data_value: edict for KITTI dataset value
	x: input layer after bi_filters
	mask: Mask for dataset
	"""
	def __init__(self, data_value, stride=1, padding=0):
		super(Recurrent, self).__init__()
		self.data_value = data_value
		self.stride = stride
		self.padding = padding

	def forward(self,x,mask,filters):
		# pdb.set_trace()
		data_dict = self.data_value
		size_z, size_a = data_dict.LCN_HEIGHT, data_dict.LCN_WIDTH
		# initialsing compatibility matrix
		compatibility_kernel_init = torch.from_numpy(np.reshape(np.ones(
			(data_dict.NUM_CLASSES, data_dict.NUM_CLASSES), dtype = "float32")- np.identity(
			data_dict.NUM_CLASSES, dtype="float32"),[data_dict.NUM_CLASSES, data_dict.NUM_CLASSES,1,1]))

		bi_compat_kernel = compatibility_kernel_init*data_dict.BI_FILTER_COEFF
		bi_compat_kernel.requires_grad_()

		angular_compat_kernel = compatibility_kernel_init* data_dict.ANG_FILTER_COEFF
		angular_compat_kernel.requires_grad_()

		condensing_kernel = torch.from_numpy(util.condensing_matrix(data_dict.NUM_CLASSES, size_z, size_a)).float()

		angular_filters = torch.from_numpy(util.angular_filter_kernel(data_dict.NUM_CLASSES, size_z, size_a, data_dict.ANG_THETA_A**2)).float()

		bi_angular_filters = torch.from_numpy(util.angular_filter_kernel(data_dict.NUM_CLASSES, size_z, size_a, data_dict.BILATERAL_THETA_A**2)).float()

		bi_compat_kernel, angular_compat_kernel, condensing_kernel, angular_filters, bi_angular_filters = bi_compat_kernel.to(device), angular_compat_kernel.to(device), condensing_kernel.to(device), angular_filters.to(device), bi_angular_filters.to(device)

		for i in range(data_dict.RCRF_ITER):
			# pdb.set_trace()
			inputs = F.softmax(x,dim=1)

			pad_z, pad_a = size_z//2, size_a//2
			half_filter_dim = (size_z*size_a)//2
			batch, in_channel, zenith, azimuth = list(inputs.size())
			ang_output = F.conv2d(inputs, weight=angular_filters, stride=self.stride, padding=self.padding)
			bi_ang_output = F.conv2d(inputs, weight=bi_angular_filters,stride=self.stride, padding=self.padding)
			condensed_input = F.conv2d(inputs*mask, weight=condensing_kernel, stride=self.stride, padding=self.padding)
			condensed_input = condensed_input.view(batch,in_channel,size_z*size_a-1,zenith,azimuth)
			condensed_input = torch.sum((condensed_input * filters), 2)
			bi_output = torch.mul(condensed_input, mask)
			bi_output *= bi_ang_output
			ang_output = F.conv2d(ang_output, weight=angular_compat_kernel, stride=self.stride, padding=0)
			bi_output = F.conv2d(bi_output, weight = bi_compat_kernel, stride=self.stride, padding=0)
			pairwise = torch.add(ang_output, bi_output)
			outputs = torch.add(inputs, pairwise)
			x = outputs

		return outputs