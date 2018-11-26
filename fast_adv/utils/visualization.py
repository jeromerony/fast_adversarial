# From https://github.com/luizgh/visdom_logger

import visdom
import torch
from collections import defaultdict


class VisdomLogger:
    def __init__(self, port):
        self.vis = visdom.Visdom(port=port)
        self.windows = defaultdict(lambda: None)

    def scalar(self, name, x, y):
        win = self.windows[name]

        update = None if win is None else 'append'
        win = self.vis.line(torch.Tensor([y]), torch.Tensor([x]),
                            win=win, update=update, opts={'legend': [name]})

        self.windows[name] = win

    def scalars(self, list_of_names, x, list_of_ys):
        name = '$'.join(list_of_names)

        win = self.windows[name]

        update = None if win is None else 'append'
        list_of_xs = [x] * len(list_of_ys)
        win = self.vis.line(torch.Tensor([list_of_ys]), torch.Tensor([list_of_xs]),
                            win=win, update=update, opts={'legend': list_of_names})

        self.windows[name] = win

    def images(self, name, images, mean_std=None):
        win = self.windows[name]

        win = self.vis.images(images if mean_std is None else
                              images * torch.Tensor(mean_std[0]) + torch.Tensor(mean_std[1]),
                              win=win, opts={'legend': [name]})

        self.windows[name] = win

    def reset_windows(self):
        self.windows.clear()
