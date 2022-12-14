import time
import logging
import warnings
import torch
from statistics import mean
import tqdm
class Trainer:
    """Trainer
    
    Class that eases the training of a PyTorch model.
    
    Parameters
    ----------
    model : torch.Module
        The model to train.
    criterion : torch.Module
        Loss function criterion.
    optimizer : torch.optim
        Optimizer to perform the parameters update.
    logger_kwards : dict
        Args for ..
        
    Attributes
    ----------
    train_loss_ : list
    val_loss_ : list
    
    """
    def __init__(self, model, criterion, optimizer, scheduler, logger_kwargs, device = None):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger_kwargs = logger_kwargs
        self.device = self._get_device(device)
        
        # send model to device
        self.model.to(self.device)

        # attributes        
        self.train_loss_ = []
        self.val_loss_ = []

        logging.basicConfig(level=logging.INFO)
        
    def fit(self, train_loader, val_loader, epochs):
        """Fits.
        
        Fit the model using the given loaders for the given number
        of epochs.
        
        Parameters
        ----------
        train_loader : 
        val_loader : 
        epochs : int
            Number of training epochs.
        
        """
        # track total training time
        total_start_time = time.time()

        # ---- train process ----
        pbar = tqdm.trange(epochs, unit="epoch")
        for epoch in pbar:
            # track epoch time
            pbar.set_description(f"Epoch: {epoch}")
            epoch_start_time = time.time()

            # train
            tr_loss = self._train(train_loader)
            
            # validate
            val_loss = self._validate(val_loader)
            self.train_loss_.append(tr_loss)
            self.val_loss_.append(val_loss)
            self.scheduler.step(val_loss)
            epoch_time = time.time() - epoch_start_time
            self._logger(
                tr_loss, 
                val_loss, 
                epoch+1, 
                epochs, 
                epoch_time 
                #**self.logger_kwargs
            )
            pbar.set_postfix(train_loss = tr_loss, val_loss = val_loss)
        total_time = time.time() - total_start_time

        # final message
        logging.info(
            f"""End of training. Total time: {round(total_time, 5)} seconds"""
        )
        
    
    def _logger(
        self, 
        tr_loss, 
        val_loss, 
        epoch, 
        epochs, 
        epoch_time, 
        show=True, 
        update_step=1
    ):
        if show:
            if epoch % update_step == 0 or epoch == 1:
                # to satisfy pep8 common limit of characters
                msg = f"Epoch {epoch}/{epochs} | Train loss: {tr_loss}" 
                msg = f"{msg} | Validation loss: {val_loss}"
                msg = f"{msg} | Learning rate: {self.optimizer.param_groups[0]['lr']}"
                msg = f"{msg} | Time/epoch: {round(epoch_time, 5)} seconds"

                logging.info(msg)
                
    
    def _train(self, loader):
        self.model.train()
        loss_ = []
        with tqdm.tqdm(loader, unit = 'batch', desc = 'Train loader') as tepoch:
            for features, ground_truth,_ ,_ in tepoch:
                # move to device
                features, ground_truth = self._to_device(features, ground_truth, self.device)
                
                # forward pass
                out = self.model(features)
                
                # loss
                loss = (self._compute_loss(out.float(), ground_truth.float()))
                loss_.append(loss.item())
                
                # remove gradient from previous passes
                self.optimizer.zero_grad()
                
                # backprop
                loss.backward()
                
                # parameters update
                self.optimizer.step()
                
                tepoch.set_postfix(loss = loss.item())
            
        return mean(loss_)
    
    def _to_device(self, features, ground_truth, device):
        return features.to(device, dtype = torch.float), ground_truth.to(device, dtype = torch.float)
    
    def _validate(self, loader):
        self.model.eval()
        loss_ = []
        with torch.no_grad():
            with tqdm.tqdm(loader, unit = 'batch', desc = 'Validation loader') as tepoch:
                for features, ground_truth, _, _ in tepoch:
                    # move to device
                    features, ground_truth = self._to_device(
                        features, 
                        ground_truth, 
                        self.device
                    )
                    
                    out = self.model(features)
                    loss = self._compute_loss(out.float(), ground_truth.float())
                    loss_.append(loss.item())
                    tepoch.set_postfix(loss = loss.item())
        return mean(loss_)
    
    def _compute_loss(self, real, target):
        try:
            loss = self.criterion(real, target)
        except:
            loss = self.criterion(real, target.long())
            msg = f"Target tensor has been casted from"
            msg = f"{msg} {type(target)} to 'long' dtype to avoid errors."
            warnings.warn(msg)

        # apply regularization if any
        # loss += penalty.item()
            
        return loss

    def _get_device(self, device):
        if device is None:
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            msg = f"Device was automatically selected: {dev}"
            warnings.warn(msg)
        else:
            dev = device

        return dev

    def save_checkpoint(path, epoch, model, optimizer):
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, path)

