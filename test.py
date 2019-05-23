import models
import torch as t
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
from data.dataset import  TrademarkImageDataSet
from Config import config as cfg
from utils.visualize import Visualizer
from torchnet import meter
from torchvision import models as tv_model
import utils.tool as tool
import numpy as np
cfg = cfg()


def val(model, dataloader):
	model.eval()
	
	confusion_matrix = meter.ConfusionMeter(cfg.class_num)
	
	for ii, data in enumerate(dataloader):
		input, label = data

		bs, ncrops, c, h, w = input.size()
		label = label.view(-1,1)
		input = input.view(-1, c, h, w)
		
		val_input = V(input, volatile=True)
		val_label = V(label - 1, volatile=True)
		
		if cfg.use_gpu:
			val_input, val_label = val_input.cuda(), val_label.cuda()
		score = model(val_input)
		
		# confusion_matrix.add(score.cpu().data.squeeze(), val_label.cpu().data.long())
		confusion_matrix.add(score.cpu().data, val_label.cpu().data.squeeze())
	
	# model.train()
	
	cm_value = confusion_matrix.value()
	sum = 0.0
	for i in range(len(cm_value)):
		sum += cm_value[i][i]
	
	accuracy = 100. * sum / (cm_value.sum())
	
	return confusion_matrix, accuracy


if __name__ == '__main__':
	
	model = tv_model.resnet50()
	model.avgpool = t.nn.AvgPool2d((7, 13), stride=1)
	num_frts = model.fc.in_features
	model.fc = t.nn.Linear(num_frts, cfg.class_num)

	# model = models.ResNet34(num_classes=100)
	
	if cfg.load_model_path:
		tool.load(model , cfg.load_model_path)
	if cfg.use_gpu:
		model.cuda()
		
	# train
	train_data = TrademarkImageDataSet(cfg.train_img_dir, cfg.train_label_path, train=False)
	train_dataloader = DataLoader(train_data, batch_size=cfg.test_batchsize, shuffle=False, num_workers=cfg.num_workers)
	print(len(train_dataloader)*cfg.batchsize)
	val_cm, val_accuracy = val(model, train_dataloader)
	print('train.accuracy:{accuracy}'.format(accuracy= val_accuracy ))
	
	# val
	val_data = TrademarkImageDataSet(cfg.train_img_dir, cfg.val_label_path, train=False)
	val_dataloader = DataLoader(val_data, batch_size=cfg.test_batchsize, shuffle=True, num_workers=cfg.num_workers)
	print(len(val_dataloader) * cfg.batchsize)
	val_cm, val_accuracy = val(model, val_dataloader)
	print('val.accuracy:{accuracy}'.format(accuracy=val_accuracy))

	test_data = TrademarkImageDataSet(cfg.test_img_dir, cfg.test_label_path, test=True)
	test_dataloader = DataLoader(test_data, batch_size=cfg.test_batchsize, shuffle=False, num_workers=cfg.num_workers)
	print(len(test_dataloader) * cfg.batchsize)
	val_cm, val_accuracy = val(model, test_dataloader)
	print('test.accuracy:{accuracy}'.format(accuracy=val_accuracy))