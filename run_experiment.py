import subprocess
import sys
import os
from train import Train

def install_requirements():
    if not os.path.exists("requirements.txt"):
        print("No se encuentra el archivo requirements.txt")
        sys.exit(1)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def train():
    Train.train_model_ebm()
    Train.train_linear_regression()
    Train.lgbmregressor()
    Train.neural_network()


def start_mlflow_ui():
    subprocess.check_call([sys.executable, "-m", "mlflow", "ui"])
if __name__ == "__main__":
    # install_requirements()
    train()
    start_mlflow_ui()
