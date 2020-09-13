import os
import torch
import random
import numpy as np
from visdom import Visdom
import torch.nn.functional as F


class AbstractCallback(object):
    def per_batch(self, args):
        raise RuntimeError("Don\'t implement batch callback method")

    def per_epoch(self, args):
        raise RuntimeError("Don\'t implement epoch callback method")

    def early_stopping(self, args):
        raise RuntimeError("Don\'t implement early stopping callback method")


class SaveModelPerEpoch(AbstractCallback):
    def __init__(self, path, save_step=1):
        self.path = path
        self.step=save_step

        if not os.path.isdir(path):
            os.makedirs(path)

    def per_batch(self, args):
        pass

    def per_epoch(self, args):
        if args['n'] % self.step == 0:
            args['model'].save(
                os.path.join(self.path, 'model-{}.trh'.format(args['n']))
            )

    def early_stopping(self, args):
        args['model'].save(
            os.path.join(self.path, 'early_model-{}.trh'.format(args['n']))
        )


class SaveOptimizerPerEpoch(AbstractCallback):
    def __init__(self, path, save_step=1):
        self.path = path
        self.step=save_step

        if not os.path.isdir(path):
            os.makedirs(path)

    def per_batch(self, args):
        pass

    def per_epoch(self, args):
        if args['n'] % self.step == 0:
            torch.save(args['optimize_state'], (
                os.path.join(
                    self.path,
                    'optimize_state-{}.trh'.format(args['n'])
                )
            ))

    def early_stopping(self, args):
        torch.save(args['optimize_state'], (
            os.path.join(
                self.path,
                'early_optimize_state-{}.trh'.format(args['n'])
            )
        ))


class VisPlot(AbstractCallback):
    def __init__(self, title, server='https://localhost', port=8080):
        self.viz = Visdom(server=server, port=port)
        self.windows = {}
        self.title = title

    def register_scatterplot(self, name, xlabel, ylabel, legend=None):
        options = dict(title=self.title, markersize=5,
                        xlabel=xlabel, ylabel=ylabel) if legend is None \
                       else dict(title=self.title, markersize=5,
                        xlabel=xlabel, ylabel=ylabel,
                        legend=legend)

        self.windows[name] = [None, options]

    def update_scatterplot(self, name, x, y1, y2=None, window_size=100):
        """
        Update plot
        Args:
            name: name of updating plot
            x: x values for plotting
            y1: y values for plotting
            y2: plot can contains two graphs
            window_size: window size for plot smoothing (by mean in window)
        Returns:
        """
        if y2 is None:
            self.windows[name][0] = self.viz.line(
                np.array([y1], dtype=np.float32),
                np.array([x], dtype=np.float32),
                win=self.windows[name][0],
                opts=self.windows[name][1],
                update='append' if self.windows[name][0] is not None else None
            )
        else:
            self.windows[name][0] = self.viz.line(
                np.array([[y1, y2]], dtype=np.float32),
                np.array([x], dtype=np.float32),
                win=self.windows[name][0],
                opts=self.windows[name][1],
                update='append' if self.windows[name][0] is not None else None
            )

    def per_batch(self, args, keyward='per_batch'):
        for win in self.windows.keys():
            if keyward in win:
                if 'train' in win:
                    self.update_scatterplot(
                        win,
                        args['n'],
                        args['loss'],
                        args['loss_xn']
                    )

                if 'validation' in win:
                    self.update_scatterplot(
                        win,
                        args['n'],
                        args['val loss'],
                        args['val acc']
                    )

    def per_epoch(self, args, keyward='per_epoch'):
        for win in self.windows.keys():
            if keyward in win:
                if 'train' in win and 'validation' in win and 'acc' not in win:
                    self.update_scatterplot(
                        win,
                        args['n'],
                        args['loss'],
                        args['val loss']
                    )

                if 'train' in win and 'validation' in win and 'acc' in win:
                    self.update_scatterplot(
                        win,
                        args['n'],
                        args['acc'],
                        args['val acc']
                    )

    def early_stopping(self, args):
        pass


class VisImageForAE(AbstractCallback):
    def __init__(self, title, server='https://localhost', port=8080,
                 vis_step=1, scale=10):
        self.viz = Visdom(server=server, port=port)
        self.title = title + ' input|original|predicted'
        self.windows = {1: None}
        self.n = 0
        self.step = vis_step
        self.scale = scale

        random.seed()

    def per_batch(self, args, label=1):
        if self.n % self.step == 0:
            i = random.randint(0, args['y_true'].size(0) - 1)

            for win in self.windows.keys():
                if win == label:
                    x = torch.cat(
                        (args['x'][i], args['y_true'][i], args['y_pred'][i]),
                        dim=2
                    )

                    self.windows[win] = self.viz.image(
                        torch.clamp(
                            F.interpolate(
                                x.unsqueeze(0),
                                scale_factor=(self.scale, self.scale)
                            ).squeeze(),
                            -0.5, 0.5
                        ) + 0.5,
                        win=self.windows[win],
                        opts=dict(title=self.title)
                    )

        self.n += 1
        if self.n >= 1000000000:
            self.n = 0

    def per_epoch(self, args):
        pass

    def early_stopping(self, args):
        pass

    def add_window(self, label):
        self.windows[label] = None