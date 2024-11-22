from loguru import logger
from loss import CrossEntropy
from models.lr import LogisticRegression1
from optim import OptimizerLg
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from trainer import Trainer


def train():
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

    logger.debug(f"Shape of X: {X.shape}, y_one_hot: {y_one_hot.shape}")
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )

    # Model setup
    n_features = X.shape[1]
    print(n_features)
    n_classes = len(iris.target_names)
    num_epochs = 20
    learning_rate = 0.1

    model = LogisticRegression1(
        n_features=n_features, n_classes=n_classes, use_bias=True
    )
    logger.debug(f"Before training | weights: {model.weights} - bias: {model.bias}")

    # Loss and optimizer
    loss_fn = CrossEntropy()
    optimizer = OptimizerLg(model=model, learning_rate=learning_rate)

    # Trainer setup and training process
    trainer = Trainer(model=model, optimizer=optimizer, epoch=num_epochs, loss=loss_fn)

    trainer.train(X_train=X_train, y_train=y_train, X_test=X_eval, y_test=y_eval)

    logger.debug(f"After training | weights: {model.weights} - bias: {model.bias}")


if __name__ == "__main__":
    train()
