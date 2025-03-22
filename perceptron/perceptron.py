entradas = [-1, 7, 5]
pesos = [0.8, 0.1, 0]


def soma(e, p):
    """
    Função que calcula a soma ponderada de duas listas
    :param e: lista de entradas
    :param p: lista de pesos
    :return: soma ponderada
    """
    s = 0
    for i in range(3):
        s += e[i] * p[i]
    return s


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
