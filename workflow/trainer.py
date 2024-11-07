from loguru import logger
from loss import Loss
from models.lr import Module

from workflow.loss import Optimizer


class Trainer:
    def __init__(self,
                 model: Module,
                 optimizer: Optimizer,
                 loss: Loss,
                 n_epochs: int):
        self.model = model
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.loss = loss

    def train(self, x_train, y_train, x_test=None, y_test=None):
        for epoch in range(self.n_epochs):
            # Step 1. Forward processing (inference)
            predicts = self.model(x_train)

            # Step 2. Calculate loss based on predicts and labels
            train_loss = self.loss(predicts, y_train)

            # Step 3. Backward processing
            # 3.1 calculate gradient descent
            dw, db = self.loss.backward(x_train)

            # 3.2 update weight and bias for model
            self.optimizer.step(dw, db)
            eval_loss = None
            if x_test is not None:
                eval_predicts = self.model(x_test)
                eval_loss = self.loss(eval_predicts, y_test)
            logger.debug(
                f"Epoch: {epoch + 1} || Training loss: {train_loss} - Evaluation loss: {eval_loss}"
            )
