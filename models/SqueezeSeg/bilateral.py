import torch
import torch.nn as nn
import torch.nn.functional as F	
import utils.util_recurrent as util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BilateralFilter(nn.Module):
	"""docstring for BilateralFilter"""
	def __init__(self, data_value, stride=1, padding=0):
		super(BilateralFilter, self).__init__()
		
		self.data_value = data_value
		self.stride = stride
		self.padding = padding

	def forward(self,x):
		data_dict = self.data_value

		batch, in_channel, zenith, azimuth = list(x.size())
		size_z, size_a = data_dict.LCN_HEIGHT, data_dict.LCN_WIDTH

		condensing_kernel = torch.from_numpy(util.condensing_matrix(in_channel, size_z, size_a)).float()
		condensed_input = F.conv2d(x, weight=condensing_kernel.to(device), stride=self.stride, padding=self.padding)
		diff_x = x[:,0,:,:].view(batch,1,zenith,azimuth) - condensed_input[:,0::in_channel,:,:]
		diff_y = x[:,1,:,:].view(batch,1,zenith,azimuth) - condensed_input[:,1::in_channel,:,:]
		diff_z = x[:,2,:,:].view(batch,1,zenith,azimuth) - condensed_input[:,2::in_channel,:,:]
		bi_filters = []

		for i in range(data_dict.NUM_CLASSES):
			theta = data_dict.BILATERAL_THETA_R[i]
			bi_filter = torch.exp(-(diff_x**2 + diff_y**2 + diff_z**2)/2/theta**2)
			bi_filters.append(bi_filter)

		bf_weight = torch.stack(bi_filters)
		bf_weight = bf_weight.transpose(0,1)

		return bf_weight