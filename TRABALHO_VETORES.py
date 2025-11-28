import matplotlib.pyplot as plt
import numpy as np

# 1. NÚCLEO MATEMÁTICO (SEM BIBLIOTECAS PRONTAS DE RESOLUÇÃO)

def resolver_sistema_gauss(A, b):
    n = len(b)
    # Cria cópia para não alterar as listas originais
    M = [linha[:] + [val] for linha, val in zip(A, b)]

    # Escalonamento
    for i in range(n):
        # Pivoteamento (achar o maior valor na coluna para evitar erros)
        pivo_idx = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > abs(M[pivo_idx][i]):
                pivo_idx = k

        M[i], M[pivo_idx] = M[pivo_idx], M[i]

        if abs(M[i][i]) < 1e-10:
            return None  # Matriz singular (não invertível)

        # Eliminação
        for k in range(i + 1, n):
            fator = M[k][i] / M[i][i]
            for j in range(i, n + 1):
                M[k][j] -= fator * M[i][j]

    # Substituição Reversa
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        soma = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - soma) / M[i][i]

    return x


def montar_matriz_vandermonde(pontos_x):
    grau = len(pontos_x)
    matriz = []
    for x in pontos_x:
        linha = [x ** i for i in range(grau)]
        matriz.append(linha)
    return matriz


def calcular_y_polinomio(x, coeficientes):
    y = 0
    for grau, coef in enumerate(coeficientes):
        y += coef * (x ** grau)
    return y


def formatar_equacao_latex(coeficientes):
    termos = []
    for grau, coef in enumerate(coeficientes):
        if abs(coef) < 1e-4: continue
        val = abs(coef)
        sinal = "+" if coef >= 0 else "-"

        if grau == 0:
            termo = f"{val:.2f}"
        elif grau == 1:
            termo = f"{val:.2f}x"
        else:
            termo = f"{val:.2f}x^{grau}"

        if not termos and coef < 0:  # Primeiro termo negativo
            termos.append(f"-{termo}")
        elif not termos:
            termos.append(f"{termo}")
        else:
            termos.append(f" {sinal} {termo}")

    return "f(x) = " + "".join(termos)