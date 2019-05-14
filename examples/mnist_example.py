import argparse
import torch
import time
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.utils import save_image

from fast_adv.models.mnist import SmallCNN
from fast_adv.attacks import DDN, CarliniWagnerL2
from fast_adv.utils import requires_grad_, l2_norm


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate adversarial examples on MNIST')
    parser.add_argument('--data-path', default='data/mnist')
    parser.add_argument('--model-path', required=True)

    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Loading data')
    dataset = datasets.MNIST(args.data_path, train=False,
                             transform=transforms.ToTensor(),
                             download=True)
    loader = data.DataLoader(dataset, shuffle=False, batch_size=16)

    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)

    print('Loading model')
    model = SmallCNN()
    model.load_state_dict(torch.load(args.model_path))
    model.eval().to(device)
    requires_grad_(model, False)

    print('Running DDN attack')
    attacker = DDN(steps=100, device=device)
    start = time.time()
    ddn_atk = attacker.attack(model, x, labels=y, targeted=False)
    ddn_time = time.time() - start

    print('Running C&W attack')
    cwattacker = CarliniWagnerL2(device=device,
                                 image_constraints=(0, 1),
                                 num_classes=10)

    start = time.time()
    cw_atk = cwattacker.attack(model, x, labels=y, targeted=False)
    cw_time = time.time() - start

    # Save images
    all_imgs = torch.cat((x, cw_atk, ddn_atk))
    save_image(all_imgs, 'images_and_attacks.png', nrow=16, pad_value=0)

    # Print metrics
    pred_orig = model(x).argmax(dim=1).cpu()
    pred_cw = model(cw_atk).argmax(dim=1).cpu()
    pred_ddn = model(ddn_atk).argmax(dim=1).cpu()
    print('Predictions on original images: {}'.format(pred_orig))
    print('Predictions on C&W attack: {}'.format(pred_cw))
    print('Predictions on DDN attack: {}'.format(pred_ddn))
    print('C&W done in {:.1f}s: Success: {:.2f}%, Mean L2: {:.4f}.'.format(
        cw_time,
        (pred_cw != y.cpu()).float().mean().item() * 100,
        l2_norm(cw_atk - x).mean().item()
    ))
    print('DDN done in {:.1f}s: Success: {:.2f}%, Mean L2: {:.4f}.'.format(
        ddn_time,
        (pred_ddn != y.cpu()).float().mean().item() * 100,
        l2_norm(ddn_atk - x).mean().item()
    ))

    print('See "images_and_attacks.png". '
          'Top: original images, mid: C&W attack; '
          'bottom: DDN attack')
