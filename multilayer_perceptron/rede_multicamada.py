import numpy as np

"""
Rede neural multicamada para simular a porta lógica AND.
Possui 2 neurônios de entrada, 3 na camada oculta e 1 de saída.
Utiliza função de ativação sigmoid e backpropagation para ajuste dos pesos.
Treinada por 100 épocas com taxa de aprendizado 0.3 e momento 1.
Implementação com NumPy.
"""


def sigmoid(soma):
    """
    Função de ativação sigmoid
    :param soma: float
    :return: float
    """
    return 1 / (1 + np.exp(-soma))


def sigmoid_derivada(sig):
    """
    Derivada da função de ativação sigmoid
    :param soma: float
    :return: float
    """
    return sig * (1 - sig)


# Entradas
entradas = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Saídas esperadas
saidas = np.array([[0], [1], [1], [0]])

# Pesos iniciais (camada de entrada para camada oculta)
# pesos0 = np.array([
#     [-0.424, -0.740, -0.961],
#     [0.358, -0.577, -0.469]
# ])

pesos0 = 2 * np.random.random((2, 3)) - 1
pesos1 = 2 * np.random.random((3, 1)) - 1

# Tempo de treinamento (training time)
epocas = 100000
taxa_aprendizado = 0.3
momento = 1

for epoca in range(epocas):

    camadaEntrada = entradas

    # Cálculo da camada oculta
    # Multiplicação de matrizes para calcular a saída da camada oculta
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)

    # Cálculo da saída
    # Multiplicação de matrizes para calcular a saída final
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)

    # Cálculo do erro
    erroCamadaSaida = saidas - camadaSaida

    # Cálcula a média absoluta do erro
    mediaAbsolutaErro = np.mean(np.abs(erroCamadaSaida))
    print(
        f'Época {epoca + 1}/{epocas} - '
        f'Erro médio absoluto: {mediaAbsolutaErro:.4f}'
    )

    # Derivada da camada de saída
    derivadaSaida = sigmoid_derivada(camadaSaida)

    # Cálculo do delta da camada de saída
    # Multiplicação do erro da camada de saída pela derivada da camada de saída
    deltaSaida = erroCamadaSaida * derivadaSaida
    # deltaOculta = np.dot(deltaSaida, pesos1.T) * \
    pesos1_transposta = pesos1.T
    delta_saidaXPeso = np.dot(deltaSaida, pesos1_transposta)
    deltaCamadaOculta = delta_saidaXPeso * sigmoid_derivada(camadaOculta)

    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = np.dot(camadaOcultaTransposta, deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxa_aprendizado)

    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxa_aprendizado)

    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxa_aprendizado)
