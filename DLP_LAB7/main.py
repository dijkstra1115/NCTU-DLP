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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--z_dim', default=100, type=int, help='')
    parser.add_argument('--c_dim', default=200, type=int, help='')
    parser.add_argument('--epochs', default=200, type=int, help='')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    # parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    # parser.add_argument('--model_dir', default='', help='base directory to save logs')
    # parser.add_argument('--data_root', default='./iclevr', help='root directory for data')
    parser.add_argument('--cuda', default=True, action='store_true')  

    args = parser.parse_args()
    return args

def train(dataloader,g_model,d_model,z_dim,epochs,lr):
    """
    :param z_dim: 100
    """
    Criterion=nn.BCELoss()
    optimizer_g=torch.optim.Adam(g_model.parameters(),lr,betas=(0.5,0.99))
    optimizer_d=torch.optim.Adam(d_model.parameters(),lr,betas=(0.5,0.99))
    model_evaluator=evaluation_model()

    test_conditions=get_test_conditions(os.path.join('dataset','test.json')).to(device)
    fixed_z = torch.randn(len(test_conditions), z_dim).to(device)
    best_score = 0

    for epoch in range(1,1+epochs):
        total_loss_g=0
        total_loss_d=0
        for i,(images,conditions) in enumerate(dataloader):
            g_model.train()
            d_model.train()
            batch_size=len(images)
            images = images.to(device)
            conditions = conditions.to(device)

            real = torch.ones(batch_size).to(device)
            fake = torch.zeros(batch_size).to(device)
            """
            train discriminator
            """
            optimizer_d.zero_grad()

            # for real images
            predicts = d_model(images, conditions)
            loss_real = Criterion(predicts, real)
            # for fake images
            z = torch.randn(batch_size, z_dim).to(device)
            # 產生fake imgae時，隨機生成latent vector但condition不是隨機生成，
            # 而是拿training data中已經有的condition，
            # 自己隨機生成的condition vector(24dim中會有1~3個1)反而會train壞掉。
            gen_imgs = g_model(z,conditions)
            # gen_imgs.detach() 是為了不要讓BP去跟新到Generator的參數
            predicts = d_model(gen_imgs.detach(), conditions)
            loss_fake = Criterion(predicts, fake)
            # bp
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            """
            train generator
            """
            for _ in range(4):
                optimizer_g.zero_grad()

                z = torch.randn(batch_size,z_dim).to(device)
                gen_imgs = g_model(z, conditions)
                predicts = d_model(gen_imgs,conditions)
                loss_g = Criterion(predicts,real)
                # bp
                loss_g.backward()
                optimizer_g.step()

            print(f'epoch{epoch} {i}/{len(dataloader)}  loss_g: {loss_g.item():.3f}  loss_d: {loss_d.item():.3f}')
            total_loss_g+=loss_g.item()
            total_loss_d+=loss_d.item()

        # evaluate
        g_model.eval()
        d_model.eval()
        with torch.no_grad():
            gen_imgs=g_model(fixed_z,test_conditions)
        score=model_evaluator.eval(gen_imgs,test_conditions)
        if score>best_score:
            best_score=score
            best_g_model_wts=copy.deepcopy(g_model.state_dict())
            best_d_model_wts=copy.deepcopy(d_model.state_dict())
            torch.save(best_g_model_wts,os.path.join('models_0614/generator',f'epoch{epoch}_score{score:.2f}.pt'))
            torch.save(best_d_model_wts,os.path.join('models_0614/discriminator',f'epoch{epoch}_score{score:.2f}.pt'))
        print(f'avg loss_g: {total_loss_g/len(dataloader):.3f}  avg_loss_d: {total_loss_d/len(dataloader):.3f}')
        print(f'testing score: {score:.2f}')
        print('---------------------------------------------')
        # savefig
        save_image(gen_imgs, os.path.join('results_0614', f'epoch{epoch}.png'), nrow=8, normalize=True)

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

if __name__=='__main__':
    args = parse_args()
    image_shape=(64,64,3)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load training data
    dataset_train=CLEVRDataset(img_path='iclevr', json_path=os.path.join('dataset','train.json'))
    loader_train=DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True,num_workers=2)

    # create generate & discriminator
    generator=Generator(args.z_dim,args.c_dim).to(device)
    discrimiator=Discriminator(image_shape,args.c_dim).to(device)
    generator.weight_init(mean=0,std=0.02)
    discrimiator.weight_init(mean=0,std=0.02)

    # train
    train(loader_train,generator,discrimiator,args.z_dim,args.epochs,args.lr)
