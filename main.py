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

cfg = cfg()

def write_csv(results,file_name):
	import csv
	with open(file_name,'w') as f:
		writer = csv.writer(f)
		writer.writerow(['id','label'])
		writer.writerows(results)
		
def test(**kwargs):
	cfg.merge_cfg(kwargs)
	
	model = getattr(models,cfg.model)().eval()
	
	if cfg.load_model_path:
		model.load( cfg.load_model_path)
	if cfg.use_gpu:
		model.cuda()
	 
	test_data = TrademarkImageDataSet(cfg.test_img_dir,cfg.test_label_path,train=False)
	test_dataloader = DataLoader(test_data,batch_size=cfg.batchsize,shuffle=False,num_workers=cfg.num_workers)

	results = []
	
	for ii,(data,path) in enumerate(test_dataloader):
		input = V(data,volatile = True)
		if cfg.use_gpu:
			input.cuda()
		score = model(input)
		probability = t.nn.functional.softmax(score)[:,1].data.tolist()
		batch_results = [(path_, probability) for path_,probability_ in zip(path,probability)]
		results += batch_results
	
	write_csv(results,cfg.result_file)
	
	return results
	

	
def val( model , dataloader ):
	model.eval()
	
	confusion_matrix = meter.ConfusionMeter(cfg.class_num)
	
	for ii,data in enumerate(dataloader):
		input,label = data
		val_input = V(input,volatile = True)
		val_label = V(label.long() - 1 ,volatile = True)
		
		if cfg.use_gpu:
			val_input,val_label  = val_input.cuda(),val_label.cuda()
		score = model(val_input)
		confusion_matrix.add(score.cpu().data.squeeze(), val_label.cpu().data.long())
		
	model.train()
	
	cm_value = confusion_matrix.value()
	sum = 0.0
	for i in range(len(cm_value)):
		sum += cm_value[i][i]
		
	accuracy = 100. * sum / (cm_value.sum())
	
	return confusion_matrix, accuracy
	
	
def train(**kwargs):
	
	# t.set_default_tensor_type('torch.cuda.FloatTensor')
	cfg.merge_cfg(kwargs)
	vis = Visualizer(cfg.env)
	
	# init complete
	# model = models.ResNet34(cfg.class_num)
	
	# fune tune resnet34
	# model = tv_model.resnet34( pretrained=True )
	# model.avgpool = t.nn.AvgPool2d((7,13), stride=1)
	# num_frts = model.fc.in_features
	# model.fc = t.nn.Linear(num_frts,cfg.class_num)
	
	# fune tune resnet50
	model = tv_model.resnet50(pretrained=True)
	
	for param in model.parameters():
		param.requires_grad = False

	model.avgpool = t.nn.AvgPool2d((7, 13), stride=1)
	num_frts = model.fc.in_features
	model.fc = t.nn.Linear(num_frts, cfg.class_num)
	
	model.train()
	
	if cfg.pre_train:
		tool.load(model,cfg.load_model_path)
	if cfg.use_gpu:
		model.cuda()
	
	train_data = TrademarkImageDataSet( cfg.train_img_dir,  cfg.train_label_path ,train= True)
	val_data = TrademarkImageDataSet(cfg.train_img_dir , cfg.val_label_path, train= False)
	val_train_data = TrademarkImageDataSet( cfg.train_img_dir,  cfg.train_label_path ,train= False)
	
	train_dataloader =  DataLoader(train_data, batch_size= cfg.batchsize,shuffle=True,num_workers=cfg.num_workers)
	val_dataloader = DataLoader(val_data , batch_size= cfg.batchsize,shuffle=False,num_workers=cfg.num_workers)
	val_train_dataloader = DataLoader(val_train_data , batch_size= cfg.batchsize,shuffle=False,num_workers=cfg.num_workers)
	
	criterion = t.nn.CrossEntropyLoss()
	
	lr = cfg.lr
	optimizer = t.optim.Adam(model.fc.parameters(),lr = lr, weight_decay= cfg.weight_deacy )
	
	# if cfg.use_gpu:
		# criterion = criterion.cuda()
		# optimizer.cuda()
		
	loss_meter = meter.AverageValueMeter()
	confusion_matrix = meter.ConfusionMeter(cfg.class_num)
	previous_loss = 1e100
	
	for epoch in range(cfg.max_epoch):
		loss_meter.reset()
		confusion_matrix.reset()
		
		for ii ,(data,label) in enumerate(train_dataloader):
			input = V( data )
			target = V( label - 1 )
			if cfg.use_gpu:
				input = input.cuda()
				target = target.cuda()
			optimizer.zero_grad()
			score = model(input)

			loss = criterion(score,target)
			loss.backward()
			optimizer.step()
			
			loss_meter.add(loss.data[0])
			confusion_matrix.add(score.data,target.data)
			
			if ii % cfg.print_freq==cfg.print_freq-1:
				vis.plot('loss',loss_meter.value()[0])
			print('epoch:{epoch} , batch:{batch}/batch_num:{batch_num},loss:{loss}'.format(epoch=epoch,batch=ii,batch_num=len(train_dataloader),loss=loss.cpu().data[0]))

		
		val_cm,val_accuracy = val(model,val_dataloader)
		vis.plot('val_accuracy',val_accuracy)
		vis.log('epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}'.format(
			epoch = epoch,
			loss = loss_meter.value()[0],
			val_cm= str(val_cm.value()),
			train_cm = str(confusion_matrix.value()),
			lr = lr
		))
		
		val_train_cm,val_train_accuracy = val(model,val_train_dataloader)
		vis.plot('val_train_accuracy',val_train_accuracy)
		
		if loss_meter.value()[0] > previous_loss:
			lr = lr * cfg.lr_deacy
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr
		
		previous_loss = loss_meter.value()[0]
		
		if epoch % 10 ==0:
			tool.save(model , epoch= epoch)
	
	tool.save(model, epoch=epoch)

if __name__== "__main__":
	# import fire
	# fire.Fire()
	train()
	
	# x = V(t.Tensor( t.rand(2,3)) )
	# if cfg.use_gpu:
	# 	x = x.cuda()
	# print(x.is_cuda)