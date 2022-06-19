import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import copy
import json

from evaluator import evaluation_model
from torchvision.utils import save_image

from dataloader import CLEVRDataset
from model import Generator, Discriminator
#----#
import argparse
import ipdb

def get_test_conditions(path):
    """
    :return: (#test conditions,#classes) tensors
    """
    with open(os.path.join('dataset', 'objects.json'), 'r') as file:
        classes = json.load(file)
    with open(path,'r') as file:
        test_conditions_list=json.load(file)

    labels=torch.zeros(len(test_conditions_list),len(classes))
    for i in range(len(test_conditions_list)):
        for condition in test_conditions_list[i]:
            labels[i,int(classes[condition])]=1.

    return labels

def test(g_model,z_dim,epochs):
    """
    :param z_dim: 100
    """
    model_evaluator=evaluation_model()

    new_test_conditions=get_test_conditions(os.path.join('dataset','new_test.json')).to(device)
    # fixed_z = torch.randn(len(new_test_conditions), z_dim).to(device)
    best_score = 0

    for epoch in range(epochs):

	    # evaluate
	    g_model.eval()
	    fixed_z = torch.randn(len(new_test_conditions), z_dim).to(device)
	    with torch.no_grad():
	        gen_imgs=g_model(fixed_z, new_test_conditions)
	    score=model_evaluator.eval(gen_imgs, new_test_conditions)
	    print(f'testing score: {score:.2f}')
	    # savefig
	    save_image(gen_imgs, os.path.join('new_test_results', f'epoch{epoch}.png'), nrow=8, normalize=True)

if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create generate & discriminator
    generator=Generator(100, 200).to(device)
    # do we need to initial the normal_weight in testing?
    generator.weight_init(mean=0,std=0.02)

    generator.load_state_dict(torch.load('./models/generator/epoch156_score0.69.pt'))

    # test
    test(generator,100,20)
