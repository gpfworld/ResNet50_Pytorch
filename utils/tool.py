import time
import torch as t
from Config import config as cfg
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cfg = cfg()

def save(model , epoch=None, name=None):
	if name is None:
		dir = 'checkpoints/' + cfg.env + '/'
		if not os.path.isdir(dir):
			os.mkdir(dir)
		prefix = dir + cfg.model + "_" + str(epoch) + '_'
		name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
	t.save(model.state_dict(), name)
	return name


def load(model, path):
	model.load_state_dict(t.load(path))



def DrawImage(img,title=None):
	if title:
		title = 'img'
	plt.figure(title)
	plt.figure(num=1, figsize=(8,5),)
	plt.title('The image title')
	plt.axis('off')
	plt.imshow(img)
	plt.show()

def imshow(inp, title=None):
	"""Imshow for Tensor."""
	inp = inp.numpy().transpose((1, 2, 0))
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	inp = std * inp + mean
	plt.imshow(inp)
	if title is not None:
		plt.title(title)
	plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ =='__main__':
	if not os.path.isdir('../checkpoints/resnet34'):
		os.mkdir('../checkpoints/resnet34')