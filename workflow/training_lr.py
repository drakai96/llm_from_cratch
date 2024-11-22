from loguru import logger
from loss import MSELoss
from models.lr import LinearRegression
from optim import Optimizer
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from trainer import Trainer


def train():
    n_features = 2
    X, y = make_regression(
        n_samples=2000, n_features=n_features, noise=1, random_state=42
    )
    logger.debug(f"Shape of X: {X.shape}, y: {y.shape}")
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2)
    breakpoint()
    num_epochs = 20
    learning_rate = 0.2
    model = LinearRegression(n_features=n_features, use_bias=True)
    logger.debug(f"Before training | weight: {model.weight} - bias: {model.bias}")
    print(model.weight)
    loss_fn = MSELoss()
    optimizer = Optimizer(model=model, learning_rate=learning_rate)

    trainer = Trainer(
        model=model, optimizer=optimizer, n_epochs=num_epochs, loss=loss_fn
    )

    trainer.train(x_train=X_train, y_train=y_train, x_test=X_eval, y_test=y_eval)

    logger.debug(f"After training | weight: {model.weight} - bias: {model.bias}")


if __name__ == "__main__":
    train()
