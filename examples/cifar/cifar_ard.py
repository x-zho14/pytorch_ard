import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import sys
sys.path.append('../')

from models import LeNetARD, vgg19_bn_new_fc
from torch_ard import get_ard_reg, get_dropped_params_ratio, ELBOLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_baseline_file = 'checkpoint/ckpt_baseline.t7'
ckpt_file = 'checkpoint/ckpt_ard.t7'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
reg_factor = 1e-5

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
model = vgg19_bn_new_fc().to(device)


if os.path.isfile(ckpt_file):
    state_dict = model.state_dict()
    checkpoint = torch.load(ckpt_file)
    state_dict.update(checkpoint['net'])
    model.load_state_dict(state_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
elif os.path.isfile(ckpt_baseline_file):
    state_dict = model.state_dict()
    checkpoint = torch.load(ckpt_baseline_file)
    state_dict.update(checkpoint['net'])
    model.load_state_dict(state_dict, strict=False)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = ELBOLoss(model, F.cross_entropy).to(device)
optimizer = optim.SGD(model.parameters(), lr=200,
                      momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
n_epoches = 200
def get_kl_weight(epoch): return min(1, 1e-4 * epoch / n_epoches)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    kl_weight = get_kl_weight(epoch)
    model.train()
    train_loss = []
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print("batch_idx", batch_idx, len(trainloader))
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets, 1, kl_weight)
        loss.backward()
        layer = 0
        for n, m in model.named_modules():
            if hasattr(m, "log_sigma2"):
                layer += 1
                if layer == 2:
                    temp1 = m.log_sigma2[0][0][0][0]
                    print("before step", batch_idx, m.log_sigma2[0][0][0][0], m.log_sigma2.grad[0][0][0][0])
                    print(m.log_sigma2)
                    break
        # scheduler.step(loss)
        optimizer.step()
        layer = 0
        for n, m in model.named_modules():
            if hasattr(m, "log_sigma2"):
                layer += 1
                if layer == 2:
                    temp2 = m.log_sigma2[0][0][0][0]
                    print("after step", batch_idx, m.log_sigma2[0][0][0][0], m.log_sigma2.grad[0][0][0][0])
                    print(m.log_sigma2)
                    break
        print("diff", temp2 - temp1)
        train_loss.append(loss.item())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Train loss: %.2f' % np.mean(train_loss))
    print('Train accuracy: %.2f%%' % (correct * 100.0 / total))


def test(epoch):
    global best_acc
    model.eval()
    test_loss = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets, 1, 0)

            test_loss.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    # Save checkpoint.
    acc = 100. * correct / total
    print('Test loss: %.2f' % np.mean(test_loss))
    print('Test accuracy: %.2f%%' % acc)
    print('Compression: %.2f%%' % (100. * get_dropped_params_ratio(model)))
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': model.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, ckpt_file)
    #     best_acc = acc

for n, m in model.named_modules():
    print(n, m)
    if hasattr(m, "log_sigma2"):
        print(f"==> print {n} 's log_sigma2 grad magnitude'")
        print(m.log_sigma2.grad)
for epoch in range(start_epoch, start_epoch + n_epoches):
    train(epoch)
    log_list = []
    log_layer_dict = {}
    idx = 0
    import matplotlib.pyplot as plt
    sum = 0
    for n, m in model.named_modules():
        if hasattr(m, "log_sigma2"):
            idx += 1
            print(f"==> flattening {n} subnet")
            log_flattened = torch.flatten(torch.abs(m.log_sigma2.grad.data))
            print("fraction of zero grad:", float(torch.sum(log_flattened==0).item())/log_flattened.nelement())
            val_flattened = torch.flatten(m.log_sigma2.data)
            print("fraction of non -10:", float(torch.sum(val_flattened != -10).item()) / val_flattened.nelement())
            sum += torch.sum(log_flattened==0).item()
            log_layer_dict[idx] = log_flattened.tolist()
            plt.hist(log_flattened.tolist(), bins=50)
            # plt.xlim(0, 1)
            plt.xlabel("Probability")
            plt.ylabel("# of Weights")
            plt.title("Histogram of Probability at Layer "+ str(idx))
            plt.grid(True, linestyle="--")
            plt.savefig("Lenet_"+str(idx)+"epoch"+str(epoch)+".pdf", bbox_inches='tight')
            plt.clf()
            plt.cla()
            log_list.extend(log_flattened.tolist())
    print("Whole fraction of zero grad:", 1 - np.count_nonzero(log_list)/len(log_list))
    print("check, ", len(log_list)- np.count_nonzero(log_list), sum)
    n, bins, patches = plt.hist(log_list, bins=50)
    # plt.xlim(0, 1)
    plt.xlabel("Grad")
    plt.ylabel("Frequecy")
    plt.title("Histogram of Grad of All Layers")
    plt.grid(True, linestyle="--")
    plt.savefig("Lenet"+"_"+'whole'+str(epoch)+".pdf", bbox_inches='tight')
    # for n, m in model.named_modules():
    #     print(n, m)
    #     if hasattr(m, "log_sigma2"):
    #         print(f"==> print {n} 's log_sigma2 grad magnitude'")
    #         print(m.log_sigma2.grad)
    test(epoch)
