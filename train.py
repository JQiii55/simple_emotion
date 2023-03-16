#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as lr_scheduler
import torchvision
from torchvision import datasets,models,transforms
from torchsummary import summary
import os 
from net import simpleconv3

# 使用tensorboardX进行可视化
from tensorboardX import SummaryWriter
writer = SummaryWriter('log') #创建一个SummaryWriter的示例，默认目录名字为runs。将训练过程中的损失、精度、梯度等信息写入日志文件，然后用tensorboard工具可视化这些信息
   
## 训练主函数
def train_model(model,criterion,optimizer,scheduler,num_epochs=25):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step() ## 打印出当前的轮数。每隔一定的轮数或步数，将学习率乘以一个系数,逐渐降低学习率，有助于收敛
                model.train(True) ## 设置为训练模式
            else:
                model.train(False) ## 设置为验证模式
            
            #记录每个轮数的训练损失、训练精度和批次数
            running_loss = 0.0 ##损失变量
            running_accs = 0.0 ##精度变量
            number_batch = 0   ##初始化批次数为0。
            
            ## 从dataloaders中获得数据
            for data in dataloaders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                
                optimizer.zero_grad() ##清空梯度
                outputs = model(inputs) ##前向运行
                _,preds = torch.max(outputs.data,1)
                loss = criterion(outputs,labels) ## 用损失函数计算输出结果和真实标签之间的差异。
                if phase == 'train':
                    loss.backward() #误差反向传播
                    optimizer.step() #参数更新
                    
                running_loss += loss.data.item()
                running_accs += torch.sum(preds==labels).item()
                number_batch += 1
            
            ## 得到每一个epoch的平均损失与精度
            epoch_loss = running_loss / number_batch 
            epoch_acc = running_accs / data_sizes[phase]    
            
            ## 收集精度和损失用于可视化
            # 如果得到的损失和精度是在训练时得到的就写入对应的标签下，否则就另一个
            if phase == 'train':
                writer.add_scalar('data/trainloss',epoch_loss,epoch)
                writer.add_scalar('data/trainacc',epoch_acc,epoch)
            else:
                writer.add_scalar('data/valloss',epoch_loss,epoch)
                writer.add_scalar('data/valacc',epoch_acc,epoch)
                
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    
    writer.close()
    return model            

if __name__=='__main__':
    image_size = 64
    crop_size = 48 # 裁剪尺寸
    nclass = 4
    model = simpleconv3(nclass)
    data_dir = './data'
    
    # 创建一个名为models的目录
    if not os.path.exists('models'):
        os.mkdir('models')
        
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    print(model)
    
    ## 创建数据预处理函数，训练预处理包括随机裁剪缩放、随机翻转、归一化，验证预处理包括中心裁剪，归一化
            
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(48,scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ]),
    'val':transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),               
    ])
}       
    
    ## 使用torchvision的dataset ImageFolder接口读取数据
    data_dir = '/root/ljq/simple_emotion/data'
    image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),
                                            data_transforms[x]) 
                      for x in['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=32,
                                                  shuffle = True,
                                                  num_workers=4) 
                   for x in ['train', 'val']}
    
    # 获得数据集大小
    data_sizes = {x:len(image_datasets[x]) for x in ['train', 'val']}
    
    # 优化目标使用交叉熵，优化方法使用带动量项的SGD，学习率迭代策略为step，每隔100个epoch，变为原来的0.1倍
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(),lr=0.1,momentum = 0.9)
    step_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft,step_size = 100,gamma = 0.1)
    
    model = train_model(model=model,
                        criterion = criterion,
                        optimizer = optimizer_ft,
                        scheduler = step_lr_scheduler,
                        num_epochs = 300)
    
    torch.save(model.state_dict(),'models/model.pt')
                            