from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as opim

class Trainer():

    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit # criterion의 약자로 loss를 의미

        super().__init__()

    def _train(self, x, y, config):
        # Turn on train mode on.
        self.model.train()

        # Suffle before begin.
        # this is starting point of each epoch
        # mini_batch로 쪼개기 전에 섞는 거임
        indices = torch.randperm(x.size(0), device=x.device)
        x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
        y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)

        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x,y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze()) # loss function으로 cross-entropy를 사용

            # 학습을 수행
            # Initialize the grandients of the model.
            self.optimizer.zero_grad()
            less_i.backward()

            self.optimizer.step()

            if config.verbose <= 2:
                print("Train Iteration(%d/%d): loss=%.4e" % (i+1, len(x), float(loss_i)))

            total_loss += float(loss_i)

        return total_loss / len(x)

    def _validate(self, x, y, config):
        # Turn evaluation mode on.
        self.model.eval()

        # Turn on the no_grad mode to make more efficintly.
        with torch.no_grad():
            #Shuffle before begin.
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
            y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)
            
            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x,y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i+1, len(x), float(loss_i)))
                total_loss += float(loss_i)

            return total_loss / len(x)


    # |train_data| = |valid_data| = [(batch_size, 784), (batch_size, 1)
    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs):
            # train_data[0] : data , train_data[1] : label
            train_loss = self._train(train_data[0], train_data[1],config)
            valid_data = slef._validate(valid_data[0], valid_data[1], config)

            # You must use deep copy to take a snapshot of current best weights.
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch_index + 1,
                config.n_epoches,
                train_loss,
                valid_loss,
                lowest_loss,
            ))

        # Restore to best model
        self.model.load_state_dict(best_model)
