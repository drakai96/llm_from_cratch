from loguru import logger
from loss import Loss
from models.nn import Module

from workflow.optim import Optimizer, OptimizerLg


# class Trainer:
#     def __init__(self, model: Module, optimizer: Optimizer, loss: Loss, n_epochs: int):
#         self.model = model
#         self.optimizer = optimizer
#         self.n_epochs = n_epochs
#         self.loss = loss

#     def train(self, x_train, y_train, x_test=None, y_test=None):
#         for epoch in range(self.n_epochs):
#             # Step 1. Forward processing (inference)
#             predicts = self.model(x_train)

#             # Step 2. Calculate loss based on predicts and labels
#             train_loss = self.loss(predicts, y_train)

#             # Step 3. Backward processing
#             # 3.1 calculate gradient descent
#             dw, db = self.loss.backward(x_train)

#             # 3.2 update weight and bias for model
#             self.optimizer.step(dw, db)
#             eval_loss = None
#             if x_test is not None:
#                 eval_predicts = self.model(x_test)
#                 eval_loss = self.loss(eval_predicts, y_test)
#             logger.debug(
#                 f"Epoch: {epoch + 1} || Training loss: {train_loss} - Evaluation loss: {eval_loss}"
#             )


class Trainer:

    def __init__(self, model: Module, optimizer: OptimizerLg, loss: Loss, epoch: int):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.epoch = epoch

    def train(self, X_train, y_train, X_test=None, y_test=None):
        for i in range(self.epoch):
            # Step 1: Forward processing Dự báo kết quả dựa trên trọng số hiện tại
            predict = self.model(x=X_train)

            # Step 2: Backward calculate loss Tính sai lầm khi dự báo
            loss_value = self.loss.forward(predicts=predict, labels=y_train)

            # Step 3: Calculate delta weight
            dw, db = self.loss.backward(x=X_train)

            # Step 4: Update weigth , cập nhật ngọc số
            self.optimizer.step(dw, db)
            loss_validate = None
            if X_test is not None and y_test is not None:
                ypred = self.model.forward(X_test)
                loss_validate = self.loss(predicts=ypred, labels=y_test)
            logger.debug(f"Epoch: {i} - Loss: {loss_value} - Validate: {loss_validate}")
