import numpy as np

"""
Rede neural de um neurônio

exemplo: porta lógica XOR

A rede neural é treinada para simular a porta lógica XOR

A porta lógica XOR é verdadeira se uma e somente uma das entradas for verdadeira

Não é possível treinar um perceptron para simular a porta lógica XOR, pois o perceptron é um classificador linear
"""


# Entradas
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Saídas esperadas
saidas = np.array([0, 1, 1, 0])

# Pesos
pesos = np.array([0.0, 0.0])

# Taxa de aprendizagem
taxa_aprendizagem = 0.1


def step_function(soma):
    """
    Função de ativação
    :param soma: soma ponderada
    :return: 1 se a soma for maior ou igual a 1, 0 caso contrário
    """
    if soma >= 1:
        return 1
    return 0


def calcula_saida(registro):
    """
    Calcula a saída da rede
    :param registro: registro de entrada
    :return: saída da rede
    """
    s = registro.dot(pesos)
    return step_function(s)


def treinar():
    """
    Treina a rede
    """
    erro_total = 1
    while erro_total != 0:
        erro_total = 0
        for i in range(len(saidas)):
            saida_calculada = calcula_saida(np.array(entradas[i]))
            erro = saidas[i] - saida_calculada
            erro_total += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxa_aprendizagem * entradas[i][j] * erro)
                print(f'Peso atualizado: {pesos[j]}')
        print(f'Erro total: {erro_total}')


if __name__ == '__main__':
    treinar()
    print('Rede neural treinada')
    print(calcula_saida(entradas[0]))
    print(calcula_saida(entradas[1]))
    print(calcula_saida(entradas[2]))
    print(calcula_saida(entradas[3]))
