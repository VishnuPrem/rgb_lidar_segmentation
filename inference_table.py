import torch
import os
import sys
from infer import load_my_state_dict,test,load_weights_serially
from infer_config import *
import pdb
from utils.dataloader import Squeeze_Seg
from torch.utils.data import DataLoader



def infer_model(data_dict,ARGS_MODEL_NAME,ARGS_INFERENCE_MODEL,loader_val):
	sys.path.append(os.path.join(ARGS_ROOT,'models',ARGS_MODEL_NAME+'/'))
	module = __import__(ARGS_MODEL_NAME)
	Network = getattr(module,"Net")
	model = Network(data_dict).cuda()                                    
	torch.set_num_threads(ARGS_NUM_WORKERS)                              
	
	if ARGS_MODEL_NAME in ['SqueezeSeg','Dual_SqueezeSeg']:
		model = load_weights_serially(model, torch.load(ARGS_INFERENCE_MODEL))
	else:
		model = load_my_state_dict(model, torch.load(ARGS_INFERENCE_MODEL))  
	
	test(model,loader_val,True)                                                          

if __name__ == "__main__":
	
	dataset_val = Squeeze_Seg(ROOT_DIR,'val',ARGS_INPUT_TYPE_1,ARGS_INPUT_TYPE_2)
	                                                    
	loader_val = DataLoader(dataset_val,num_workers = ARGS_NUM_WORKERS,      
				batch_size = ARGS_VAL_BATCH_SIZE, shuffle = False)                     
	
	ARGS_MODEL_NAME = 'SqueezeSeg'                                                              
	ARGS_MODEL = 'ss_XYZDI/'                                                                  
	ARGS_INFERENCE_MODEL = ARGS_ROOT + 'final_saved_models/' + ARGS_MODEL + 'model_best.pth' 
	print('[Network Name:]',ARGS_MODEL_NAME)                                                 
	infer_model(data_dict,ARGS_MODEL_NAME,ARGS_INFERENCE_MODEL,loader_val)                   

	ARGS_MODEL_NAME = 'Dual_SqueezeSeg'                                                              
	ARGS_MODEL = 'ss_XYZDI_DIRGB/'                                                                  
	ARGS_INFERENCE_MODEL = ARGS_ROOT + 'final_saved_models/' + ARGS_MODEL + 'model_best.pth' 
	print('[Network Name:]', ARGS_MODEL_NAME)                                                 
	infer_model(data_dict,ARGS_MODEL_NAME,ARGS_INFERENCE_MODEL,loader_val)                   

	ARGS_MODEL_NAME = 'resunet'                                                               
	ARGS_MODEL = 'resunet/'                                                                   
	ARGS_INFERENCE_MODEL = ARGS_ROOT + 'final_saved_models/' + ARGS_MODEL + 'model_best.pth'
	print('[Network Name:]',ARGS_MODEL_NAME)
	infer_model(data_dict,ARGS_MODEL_NAME,ARGS_INFERENCE_MODEL,loader_val)
	
	ARGS_MODEL_NAME = 'mobileunet'                                                              
	ARGS_MODEL = 'mobileunet/'                                                                  
	ARGS_INFERENCE_MODEL = ARGS_ROOT + 'final_saved_models/' + ARGS_MODEL + 'model_best.pth' 
	print('[Network Name:]',ARGS_MODEL_NAME)                                                 
	infer_model(data_dict,ARGS_MODEL_NAME,ARGS_INFERENCE_MODEL,loader_val)                   

	ARGS_MODEL_NAME = 'efficientunet'                                                              
	ARGS_MODEL = 'effnet/'                                                                  
	ARGS_INFERENCE_MODEL = ARGS_ROOT + 'final_saved_models/' + ARGS_MODEL + 'model_best.pth' 
	print('[Network Name:]',ARGS_MODEL_NAME)                                                 
	infer_model(data_dict,ARGS_MODEL_NAME,ARGS_INFERENCE_MODEL,loader_val)                   

	ARGS_MODEL_NAME = 'resfcnnet'                                                              
	ARGS_MODEL = 'resnet_fcn_1/'                                                                  
	ARGS_INFERENCE_MODEL = ARGS_ROOT + 'final_saved_models/' + ARGS_MODEL + 'model_best.pth' 
	print('[Network Name:]',ARGS_MODEL_NAME)                                                 
	infer_model(data_dict,ARGS_MODEL_NAME,ARGS_INFERENCE_MODEL,loader_val)                   

	ARGS_MODEL_NAME = 'efficientnetb0'                                                              
	ARGS_MODEL = 'effnet_b0_1/'                                                                  
	ARGS_INFERENCE_MODEL = ARGS_ROOT + 'final_saved_models/' + ARGS_MODEL + 'model_best.pth' 
	print('[Network Name:]',ARGS_MODEL_NAME)                                                 
	infer_model(data_dict,ARGS_MODEL_NAME,ARGS_INFERENCE_MODEL,loader_val)                   

	ARGS_MODEL_NAME = 'fcn32'
	ARGS_MODEL = 'fcn_8/'
	ARGS_INFERENCE_MODEL = ARGS_ROOT + 'final_saved_models/' + ARGS_MODEL + 'model_best.pth'
	ARGS_INPUT_TYPE_1='XYZDIRGB'
	print('[Network Name:]', ARGS_MODEL_NAME)
	dataset_val = Squeeze_Seg(ROOT_DIR,'val',ARGS_INPUT_TYPE_1,ARGS_INPUT_TYPE_2)  
	loader_val = DataLoader(dataset_val,num_workers = ARGS_NUM_WORKERS,            
	                        batch_size = ARGS_VAL_BATCH_SIZE, shuffle = False)     
	infer_model(data_dict,ARGS_MODEL_NAME,ARGS_INFERENCE_MODEL,loader_val)

	ARGS_INPUT_TYPE_1='XYZDIRGB'
	dataset_val = Squeeze_Seg(ROOT_DIR,'val',ARGS_INPUT_TYPE_1,ARGS_INPUT_TYPE_2)
	                                                    
	loader_val = DataLoader(dataset_val,num_workers = ARGS_NUM_WORKERS,      
				batch_size = ARGS_VAL_BATCH_SIZE, shuffle = False)                     
	data_dict.CHANNELS = 'XYZDIRGB'
	ARGS_MODEL_NAME = 'SqueezeSeg'                                                              
	ARGS_MODEL = 'ss_XYZDIRGB/'                                                                  
	ARGS_INFERENCE_MODEL = ARGS_ROOT + 'final_saved_models/' + ARGS_MODEL + 'model_best.pth' 
	print('[Network Name:]','SqueezeSeg XYZDIRGB')                                                 
	infer_model(data_dict,ARGS_MODEL_NAME,ARGS_INFERENCE_MODEL,loader_val)                   


