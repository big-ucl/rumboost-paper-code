import numpy as np
import pandas as pd
import timeit
import pickle
import torch
import torch.nn as nn
import yaml

Floatt = torch.float64

class Logit(object):
    def __init__(self, input, choice, n_vars, n_choices, beta=None, asc=None, device='cpu'):
        """Initialize the Logit class.
        
        Args:
            input (TensorVariable)
            choice (TensorVariable)
            n_vars (int): number of input variables.
            n_choices (int): number of choice alternatives.
        """

        self.input = input
        self.choice = choice

        self.device = device
        
        #define initial value for asc parameter and parameters associated to explanatory variables
        asc_init = torch.zeros((n_choices,), dtype = Floatt, device=self.device)
        if asc is None:
            asc = nn.Parameter(asc_init)
        self.asc = asc
        
        beta_init = torch.zeros((n_vars, n_choices), dtype = Floatt, device=self.device)
        if beta is None:
            beta = nn.Parameter(beta_init)
        self.beta = beta
        
        self.params = [self.beta, self.asc]
        
        #compute the utility function and the probability  of each alternative
        pre_softmax = torch.sum(input * self.beta[None, :, :], dim=1) + self.asc
        
        self.output = nn.functional.softmax(pre_softmax, dim=1)
        
        self.output_pred = torch.argmax(self.output, dim=1)
    
    def prob_choice(self):
        return self.output
    
    def prediction(self):
        return self.output_pred
        
    def errors(self, y):
        """returns the number of errors in the minibatch for computing the accuracy of model.
        
         Args:
            y (TensorVariable):  the correct label. 
        """
        if y.ndim != self.output_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.output_pred',
                ('y', y.type, 'y_pred', self.output_pred_logit.type)
            )
        if y.dtype in [torch.int16, torch.int32, torch.int64]:
            not_equal = torch.ne(self.output_pred, y)
            return not_equal.sum().float() / not_equal.numel()
        else:
            raise NotImplementedError()


class ResNetLayer(nn.Module):
    def __init__(self, n_in, n_out, device='cpu'):
        """Initialize the ResNetLayer class.
        
        Args:
            input (TensorVariable)
            n_in (int): dimensionality of input.
            n_out (int): dimensionality of output.
        """
        super(ResNetLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        
        # define Initial value of residual layer weights
        W_init = torch.eye(self.n_out, dtype=Floatt, device=device)
        
        # learnable parameters of a model 
        self.W = nn.Parameter(W_init)
        self.params = [self.W]

    def forward(self, x):
        """ return the output of each residual layer.
        
         Args:
             x (TensorVariable):  input of each residual layer. 
        """
        self.lin_output = torch.matmul(x, self.W)

        output = x - nn.functional.softplus(self.lin_output)
        return output



class ResNet(nn.Module):
    def __init__(self, n_in, n_out, n_layers=16, dropout_rate=0.2, device='cpu'):
        """Initialize the ResNet architecture.

        Args:
            n_in (int): dimensionality of input.
            n_out (int): dimensionality of output.
            n_layers (int): number of residual layers.
            dropout_rate (float): dropout rate for the fully connected layers.
        """
        super(ResNet, self).__init__()

        self.device = device
        
        self.n_layers = n_layers
        
        # define n_layers residual layer
        self.layers = nn.ModuleList([ResNetLayer(n_in, n_out, device=self.device) for _ in range(n_layers)])
        
        # define dropout layer
        #self.dropout = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(n_layers)])
        
        # define batch normalization layer
        #self.batch_norm = nn.ModuleList([nn.BatchNorm1d(n_out, dtype=Floatt, device=self.device) for _ in range(n_layers)])
        
    def forward(self, x):
        """ return the final output of ResNet architecture.
        
         Args:
             x (TensorVariable):  input of first residual layer. 
        """
        out = x
        for i in range(self.n_layers):
            out = self.layers[i](out)
            #out = self.dropout[i](out)
            #out = self.batch_norm[i](out)
        return out

class ResLogit(Logit):
    def __init__(self, input, choice, n_vars, n_choices, n_layers=16, dropout_rate = 0.2, loss_fn = nn.CrossEntropyLoss(), epochs=200, batch_size=264, device='cpu'):
        """Initialize the ResLogit class.
        
        Args:
        input (TensorVariable)
        choice (TensorVariable) : actual label
        n_vars (int): number of input variables.
        n_choices (int): number of choice alternatives.
        n_layers (int): number of residual layers.
        """
        #super(ResLogit, self).__init__()
        self.device = device
        if isinstance(input, (pd.DataFrame, pd.Series)):
            input = torch.as_tensor(input.values, dtype=Floatt, device=self.device)
        else:
            input = torch.as_tensor(input, dtype=Floatt, device=self.device)
        if isinstance(choice, (pd.DataFrame, pd.Series)):
            choice = torch.as_tensor(choice.values, dtype=torch.long, device=self.device)
        else:
            choice = torch.as_tensor(choice, dtype=torch.long, device=self.device)
        Logit.__init__(self, input, choice, n_vars, n_choices, device=self.device)
        
        self.n_vars = n_vars
        self.n_choices = n_choices
        self.n_layers = n_layers
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size

        #define the ResNet architecture.
        self.resnet_layer = ResNet(self.n_choices, self.n_choices, n_layers=n_layers, dropout_rate=dropout_rate, device=self.device)
        for i in range(self.n_layers):
            self.params.extend(self.resnet_layer.layers[i].params)

        #self.optimizer = torch.optim.RMSprop(self.params,lr=0.001, alpha=0.9, eps=1e-10, weight_decay=0)
        self.optimizer = torch.optim.Adam(self.params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        #self.optimizer = torch.optim.SGD(self.params, lr=0.001, momentum=0.9)
        
    def fit(self, input):
        
        self.input = input
        assert self.n_layers >= 1 
        
        resnet_input = torch.sum(input * self.beta[None, :, :], dim=1)

        output_resnet = self.resnet_layer.forward(resnet_input)
               
        pre_softmax = output_resnet + self.asc

        self.V = pre_softmax
        
        self.output = nn.functional.softmax(pre_softmax, dim =1)
        
        self.output_pred = torch.argmax(self.output, dim =1)

    def predict_validate(self, inputs):
        with torch.no_grad():
            for res_lay in self.resnet_layer.layers:
                res_lay.eval()
            inputs = torch.as_tensor(inputs.values, dtype=Floatt, device=self.device) if isinstance(inputs, pd.DataFrame) else torch.as_tensor(inputs, dtype=Floatt, device=self.device)
            self.fit(inputs)
            for res_lay in self.resnet_layer.layers:
                res_lay.train()
        
            return self.output.cpu().detach().numpy()

    def train(self, X_train, y_train, X_test, y_test, valid_iter = 10):
        best_loss = 1e6
        opt_epoch = self.epochs
        n_train_batches = X_train.shape[0] // self.batch_size
        no_improvement = 0
        # convert to Pytorch tensor 
        train_x_tensor = torch.as_tensor(X_train.values, dtype=Floatt, device=self.device) if isinstance(X_train, pd.DataFrame) else torch.as_tensor(X_train, dtype=Floatt, device=self.device)
        train_y_tensor = torch.as_tensor(y_train.values, dtype=torch.long, device=self.device)
        if X_test is not None:
            valid_x_tensor = torch.as_tensor(X_test.values, dtype=Floatt, device=self.device) if isinstance(X_test, pd.DataFrame) else torch.as_tensor(X_test, dtype=Floatt, device=self.device)
            valid_y_tensor = torch.as_tensor(y_test.values, dtype=torch.long, device=self.device)
        for i in range(self.epochs):
            # Shuffle the dataset at each epoch
            indices = torch.randperm(train_x_tensor.size(0))
            train_x_tensor = train_x_tensor[indices]
            train_y_tensor = train_y_tensor[indices]

            train_loss = 0
            
            for b in range(n_train_batches):
                inputs = train_x_tensor[b * self.batch_size : (b+1) * self.batch_size]
                choice = train_y_tensor[b * self.batch_size : (b+1) * self.batch_size]
                
                self.optimizer.zero_grad()
                self.fit(inputs)
                loss = self.loss_fn(self.V, choice)
                loss.backward()
                self.optimizer.step()

                train_loss += self.batch_size * loss.item()

            if X_test is not None and i % valid_iter == 0:
                
                n_valid_batches = X_test.shape[0] // self.batch_size
                indices = torch.randperm(valid_x_tensor.size(0))
                valid_x_tensor = valid_x_tensor[indices]
                valid_y_tensor = valid_y_tensor[indices]

                valid_loss = 0

                for b in range(n_valid_batches):
                    inputs = valid_x_tensor[b * self.batch_size : (b+1) * self.batch_size]
                    choice = valid_y_tensor[b * self.batch_size : (b+1) * self.batch_size]

                    with torch.no_grad():
                        for res_lay in self.resnet_layer.layers:
                            res_lay.eval()
                        self.fit(inputs)
                        v_loss = self.loss_fn(self.V, choice)
                        for res_lay in self.resnet_layer.layers:
                            res_lay.train()
                        
                        valid_loss += v_loss.item() * self.batch_size
                print(f"Epoch {i+1}: valid_loss = {valid_loss/(n_valid_batches*self.batch_size)}" + f"\n          train_loss = {train_loss/(n_train_batches*self.batch_size)}")
                if valid_loss/(n_valid_batches*self.batch_size) < best_loss:
                    best_loss = valid_loss/(n_valid_batches*self.batch_size)
                    best_loss_it = i
                if i - best_loss_it > np.maximum(2 * valid_iter, 10):
                    opt_epoch = i+1
                    break

        return best_loss, opt_epoch, self.output.cpu().detach().numpy()




    