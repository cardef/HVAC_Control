import time
import logging
import warnings


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
        self.schduler = scheduler
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
        for epoch in range(epochs):
            # track epoch time
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
                epoch_time, 
                **self.logger_kwargs
            )

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
        update_step=20
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
        for features, ground_truth in loader:
            # move to device
            features, labels = self._to_device(features, ground_truth, self.device)
            
            # forward pass
            out = self.model(features)
            
            # loss
            loss = (self._compute_loss(out, ground_truth))
            loss_.append(loss.item())
            
            # remove gradient from previous passes
            self.optimizer.zero_grad()
            
            # backprop
            loss.backward()
            
            # parameters update
            self.optimizer.step()
            
        return loss_.mean()
    
    def _to_device(self, features, ground_truth, device):
        return features.to(device), ground_truth.to(device)
    
    def _validate(self, loader):
        self.model.eval()
        loss_ = []
        with torch.no_grad():
            for features, ground_truth in loader:
                # move to device
                features, ground_truth = self._to_device(
                    features, 
                    ground_truth, 
                    self.device
                )
                
                out = self.model(features)
                loss = self._compute_loss(out, ground_truth)
                loss_.append(loss.item())
        return loss_.mean()
    
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