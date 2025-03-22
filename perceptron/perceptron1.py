import numpy as np

# Entradas
entradas = np.array([1, 7, 5])
pesos = np.array([0.8, 0.1, 0])


def soma(e, p):
    """
    Função que calcula a soma ponderada de duas listas
    :param e: lista de entradas (np.array)
    :param p: lista de pesos (np.array)
    :return: soma ponderada
    """
    return np.dot(e, p)


s = soma(entradas, pesos)


def step_function(soma):
    """
    Função de ativação
    :param soma: soma ponderada
    :return: 1 se a soma for maior ou igual a 1, 0 caso contrário
    """
    if soma >= 1:
        return 1
    return 0


r = step_function(s)
print(r)
