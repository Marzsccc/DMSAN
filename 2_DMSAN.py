# SetTime : 2021/7/23 20:34 
# Coding : utf-8 
# Author : marzsccc
# Mail : marzsccc@163.com


from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import data_loader
import ResNet as models
from Weight import Weight
from Config import *
import time
import os
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id

cuda = not no_cuda and torch.cuda.is_available()

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

source_loader = data_loader.load_training(root_path, source_name, batch_size, kwargs)
target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print(path + ' Directory created successfully.')
        pass
    else:
        print(path + ' Directory already exists.')
        pass


def mixup(root_path1, img_name1,root_path2, img_name2):
    img1 = Image.open(os.path.join(root_path1, img_name1))
    img1 = img1.convert('RGB')
    img2 = Image.open(os.path.join(root_path2, img_name2))
    img2 = img2.convert('RGB')
    img2 = img2.resize(img1.size)
    img_mixup = Image.blend(img1, img2, 0.4) #  out = image1 * (1.0 - alpha) + image2 * alpha

    return img_mixup


def createImage(imageDir1, imageDir2, saveDir):
    i = 0
    for name1, name2 in zip(os.listdir(imageDir1), os.listdir(imageDir2)):
            i = i + 1
            saveName = "mixup_" + str(i) + ".jpg"
            saveImage = mixup(root_path1=imageDir1, img_name1=name1, root_path2=imageDir2, img_name2=name2)
            saveImage.save(os.path.join(saveDir, saveName))


def mixed():
    source_list1 = ['cassava',  'grape',  'strawberry_leaf']
    source_list2 = ['cassava', 'grape', 'strawberry_leaf']
    Subdomain_list = ['1_slight',  '2_common',  '3_severe']
    img_root_path = "./dataset/plant_disease_3class_1800/"
    save_root_path = "./dataset/plant_disease_3class_1800/"
    for source1 in source_list1:
        for source2 in source_list2:
            if source1 == source2:
                continue
            if source1 != source2:
                for Subdomain in Subdomain_list:
                    mixdir = source1 + '-' + source2
                    mkExists = os.path.exists(os.path.join(save_root_path, mixdir, Subdomain))
                    if not mkExists:
                        mkdir(os.path.join(save_root_path, mixdir))
                        mkdir(os.path.join(save_root_path, mixdir, Subdomain))
                        imageDir1 = os.path.join(img_root_path, source1, Subdomain)
                        imageDir2 = os.path.join(img_root_path, source2, Subdomain)
                        saveDir = os.path.join(save_root_path, mixdir, Subdomain)   # mixed domain path
                        createImage(imageDir1, imageDir2, saveDir)
                        print('{a}-{b}-{c}-mixup---Success!'.format(a=source1, b=source2, c=Subdomain))

def train(epoch, model):
    LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    print('learning rate{: .4f}'.format(LEARNING_RATE) )
    if bottle_neck:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.bottle.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
            ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

    model.train()

    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len_source_loader
    for i in range(1, num_iter):
        data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()
        if i % len_target_loader == 0:
            iter_target = iter(target_train_loader)
        if cuda:
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()
        data_source, label_source = Variable(data_source), Variable(label_source)
        data_target = Variable(data_target)

        optimizer.zero_grad()
        label_source_pred, loss_mmd = model(data_source, data_target, label_source)
        loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source)
        lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
        loss = loss_cls + param * lambd * loss_mmd
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                epoch, i * len(data_source), len_source_dataset,
                100. * i / len_source_loader, loss.item(), loss_cls.item(), loss_mmd.item()))

def ceshi(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            s_output, t_output = model(data, data, target)
            test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target).item() # sum up batch loss
            pred = s_output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len_target_dataset
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            target_name, test_loss, correct, len_target_dataset,
            100. * correct / len_target_dataset))
    return correct


if __name__ == '__main__':
    mixed()
    target_domain_namelist = ["strawberry_leaf", "cassava", "grape"]
    mixed_domain_namelist = ["strawberry_leaf-cassava", "strawberry_leaf-grape", "cassava-strawberry_leaf", "cassava-grape", "grape-strawberry_leaf", "grape-cassava"]
    for source_name in mixed_domain_namelist:
        source_name = source_name
        for target_name in target_domain_namelist:
            if source_name.split('-')[1] != target_name:
                continue
            if source_name.split('-')[1] == target_name:
                target_name = target_name
                model = models.DMSAN(num_classes=class_num)
                correct = 0
                print(model)
                if cuda:
                    model.cuda()
                for epoch in range(1, epochs + 1):
                    train(epoch, model)
                    t_correct = ceshi(model)
                    if t_correct > correct:
                        correct = t_correct
                        best_acc = 100. * correct / len_target_dataset
                        torch.save(model, './model_save/epoch_{b}_train_{c}_val_{d}_acc_{e:.2f}.pth'.format(b=epoch,c=source_name, d=target_name, e=best_acc))
                    print('source- {} to target- {} max correct- {} max accuracy{: .2f}%\n'.format(
                          source_name, target_name, correct, 100. * correct / len_target_dataset))
