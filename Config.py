import warnings

class config(object):
	env = 'torchvision_resnet50_pretrain_funtune_fc'
	
	model = 'ResNet50'
	
	train_img_dir = '/media/gpf/Data/Code/CV/vgg/datas/train'
	train_label_path = '/media/gpf/Data/Code/CV/vgg/datas/train.txt'
	val_label_path = '/media/gpf/Data/Code/CV/vgg/datas/val.txt'
	train_val_label_path = '/media/gpf/Data/Code/CV/vgg/datas/trainval.txt'
	test_img_dir = '/media/gpf/Data/Code/CV/vgg/datas/test'
	test_label_path = '/media/gpf/Data/Code/CV/vgg/datas/test_gt.txt'
	
	pre_train = False
	
	load_model_path = 'checkpoints/ResNet50_79_0522_20:07:00.pth'

	class_num = 100
	
	batchsize = 32
	test_batchsize = 8
	use_gpu = True
	num_workers = 8
	print_freq = 20
	
	debug_file = 'tmp/debug'
	result_file = 'result.csv'
	
	max_epoch = 80
	lr = 0.0001
	lr_deacy = 0.9
	weight_deacy = 1e-4
	
	def merge_cfg(self,kwargs):
		for k,v in kwargs.items():
			if not hasattr(self, k):
				warnings.warn("warning: opt has not attribute %s" % k)
			setattr(self,k,v)
		print('user config:')
		for k,v in self.__class__.__dict__.items():
			if not k.startswith('__'):
				print(k,getattr(self,k))

