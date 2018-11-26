import os
import argparse
import tqdm

import torch
import torch.nn.functional as F
from torch.utils import data
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn

from torchvision import transforms
from torchvision.datasets import MNIST

from fast_adv.models.mnist import SmallCNN
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, VisdomLogger
from fast_adv.attacks import DDN

parser = argparse.ArgumentParser(description='MNIST Training against DDN Attack')

parser.add_argument('--data', default='data/mnist', help='path to dataset')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
parser.add_argument('--cpu', dest='cpu', action='store_true', help='force training on cpu')
parser.add_argument('--save-folder', '--sf', default='', help='folder to save state dicts')
parser.add_argument('--save-name', '--sn', default='mnist', help='name for saving the final state dict')
parser.add_argument('--save-freq', '--sfr', type=int, help='save frequency')

parser.add_argument('--batch-size', '-b', default=128, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--lr-decay', '--lrd', default=0.1, type=float, help='decay for learning rate')
parser.add_argument('--lr-step', '--lrs', type=int, help='step size for learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float, help='weight decay')
parser.add_argument('--drop', default=0.5, type=float, help='dropout rate of the classifier')

parser.add_argument('--adv', type=int, help='epoch to start training with adversarial images')
parser.add_argument('--max-norm', type=float, help='max norm for the adversarial perturbations')
parser.add_argument('--steps', default=100, type=int, help='number of steps for the attack')

parser.add_argument('--visdom-port', '--vp', type=int, help='For visualization, which port visdom is running.')
parser.add_argument('--print-freq', '--pf', default=10, type=int, help='print frequency')

args = parser.parse_args()
print(args)
if args.lr_step is None: args.lr_step = args.epochs

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
CALLBACK = VisdomLogger(port=args.visdom_port) if args.visdom_port else None

if not os.path.exists(args.save_folder) and args.save_folder:
    os.makedirs(args.save_folder)

transform = transforms.Compose([transforms.ToTensor()])

train_data = MNIST(args.data, train=True, transform=transform, download=True)
train_set = data.Subset(train_data, list(range(55000)))
val_set = data.Subset(train_data, list(range(55000, 60000)))
test_set = MNIST(args.data, train=False, transform=transform, download=True)

train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                               drop_last=True, pin_memory=True)
val_loader = data.DataLoader(val_set, batch_size=100, shuffle=True, num_workers=args.workers, pin_memory=True)
test_loader = data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=args.workers, pin_memory=True)

model = SmallCNN(drop=args.drop).to(DEVICE)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)

attacker = DDN(steps=args.steps, device=DEVICE)

best_acc = 0
best_epoch = 0

for epoch in range(args.epochs):
    cudnn.benchmark = True
    model.train()
    requires_grad_(model, True)
    accs = AverageMeter()
    losses = AverageMeter()
    attack_norms = AverageMeter()

    scheduler.step()
    length = len(train_loader)
    for i, (images, labels) in enumerate(tqdm.tqdm(train_loader, ncols=80)):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if args.adv is not None and epoch >= args.adv:
            model.eval()
            requires_grad_(model, False)
            with torch.no_grad():
                accs.append((model(images).argmax(1) == labels).float().mean().item())
            adv = attacker.attack(model, images, labels)
            l2_norms = (adv - images).view(args.batch_size, -1).norm(2, 1)
            mean_norm = l2_norms.mean()
            if args.max_norm:
                adv = torch.renorm(adv - images, p=2, dim=0, maxnorm=args.max_norm) + images
            attack_norms.append(mean_norm.item())
            requires_grad_(model, True)
            model.train()
            logits = model(adv.detach())
        else:
            logits = model(images)
            accs.append((logits.argmax(1) == labels).float().mean().item())

        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if CALLBACK and not ((i + 1) % args.print_freq):
            CALLBACK.scalar('Tr_Loss', epoch + i / length, losses.last_avg)
            CALLBACK.scalar('Tr_Acc', epoch + i / length, accs.last_avg)
            if args.adv is not None and epoch >= args.adv:
                CALLBACK.scalar('L2', epoch + i / length, attack_norms.last_avg)

    print('Epoch {} | Training | Loss: {:.4f}, Accs: {:.4f}'.format(epoch, losses.avg, accs.avg))

    cudnn.benchmark = False
    model.eval()
    requires_grad_(model, False)
    val_accs = AverageMeter()
    val_losses = AverageMeter()

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm.tqdm(val_loader, ncols=80)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            val_accs.append((logits.argmax(1) == labels).float().mean().item())
            val_losses.append(loss.item())

    if CALLBACK:
        CALLBACK.scalar('Val_Loss', epoch + 1, val_losses.last_avg)
        CALLBACK.scalar('Val_Acc', epoch + 1, val_accs.last_avg)

    print('Epoch {} | Validation | Loss: {:.4f}, Accs: {:.4f}'.format(epoch, val_losses.avg, val_accs.avg))

    if args.adv is None and val_accs.avg >= best_acc:
        best_acc = val_accs.avg
        best_epoch = epoch
        best_dict = model.state_dict()

    if args.save_freq and not (epoch + 1) % args.save_freq:
        save_checkpoint(
            model.state_dict(), os.path.join(args.save_folder, args.save_name + '_{}.pth'.format(epoch + 1)), cpu=True)

if args.adv is None:
    model.load_state_dict(best_dict)

test_accs = AverageMeter()
test_losses = AverageMeter()

with torch.no_grad():
    for i, (images, labels) in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        test_accs.append((logits.argmax(1) == labels).float().mean().item())
        test_losses.append(loss.item())

if args.adv is not None:
    print('\nTest accuracy with final model: {:.4f} with loss: {:.4f}'.format(test_accs.avg, test_losses.avg))
else:
    print('\nTest accuracy with model from epoch {}: {:.4f} with loss: {:.4f}'.format(best_epoch, test_accs.avg,
                                                                                      test_losses.avg))

print('\nSaving model...')
save_checkpoint(model.state_dict(), os.path.join(args.save_folder, args.save_name + '.pth'), cpu=True)
