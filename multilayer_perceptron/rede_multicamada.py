import numpy as np


def sigmoid(soma):
    """
    Função de ativação sigmoid
    :param soma: float
    :return: float
    """
    return 1 / (1 + np.exp(-soma))
