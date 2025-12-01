import matplotlib.pyplot as plt
import numpy as np

# 1. núcleo matemático

def resolver_sistema_gauss(A, b):
    """
    Resolve um sistema linear Ax = b utilizando Eliminação Gaussiana com Pivoteamento Parcial.
    Parâmetros:
        A: Matriz de coeficientes (nxn).
        b: Vetor de termos independentes (tamanho n).
    Retorna:
        Vetor solução x, ou None se a matriz for singular (não tiver solução única).
    """
    
    n = len(b)
    # Cria cópia para não alterar as listas originais
    M = [linha[:] + [val] for linha, val in zip(A, b)]

    # Escalonamento
    for i in range(n):
        # Pivoteamento Parcial:
        pivo_idx = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > abs(M[pivo_idx][i]): # Verifica se o pivô é zero ou muito próximo de zero, indicando sistema sem solução única
                pivo_idx = k

        M[i], M[pivo_idx] = M[pivo_idx], M[i]

        if abs(M[i][i]) < 1e-10:
            return None  # Matriz singular (não invertível)

        # Eliminação
        for k in range(i + 1, n): # Zera os elementos abaixo do pivô na coluna atual
            fator = M[k][i] / M[i][i]
            for j in range(i, n + 1): # Operação elementar: Linha k = Linha k - (fator * Linha i)
                M[k][j] -= fator * M[i][j]

    # Substituição Reversa
    x = [0.0] * n
    for i in range(n - 1, -1, -1): # Começa da última linha e sobe até a primeira, isolando o x
        soma = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - soma) / M[i][i]

    return x


def montar_matriz_vandermonde(pontos_x):
    """
    Constrói a Matriz de Vandermonde para Interpolação Polinomial.
    Cada linha i contém [x_i^0, x_i^1, ..., x_i^n].
    """
    grau = len(pontos_x)
    matriz = []
    for x in pontos_x:
        linha = [x ** i for i in range(grau)]
        matriz.append(linha)
    return matriz


def calcular_y_polinomio(x, coeficientes):
    """
    Calcula o valor de f(x) dado um conjunto de coeficientes do polinômio.
    f(x) = c0 + c1*x + c2*x^2 + ...
    """
    y = 0
    for grau, coef in enumerate(coeficientes):
        y += coef * (x ** grau)
    return y


def formatar_equacao_latex(coeficientes):
    """
    Transforma a lista de coeficientes em uma string formatada e legível 
    para ser exibida no gráfico.
    """
    termos = []
    for grau, coef in enumerate(coeficientes):
        if abs(coef) < 1e-4: continue # Ignora termos muito próximos de zero para limpar a visualização
        val = abs(coef)
        sinal = "+" if coef >= 0 else "-"

        # Formatação condicional baseada no grau do termo
        if grau == 0:
            termo = f"{val:.2f}"
        elif grau == 1:
            termo = f"{val:.2f}x"
        else:
            termo = f"{val:.2f}x^{grau}"

        # Lógica para montar a string com os sinais corretos
        if not termos and coef < 0:  # Primeiro termo negativo
            termos.append(f"-{termo}")
        elif not termos:
            termos.append(f"{termo}")
        else:
            termos.append(f" {sinal} {termo}")

    return "f(x) = " + "".join(termos)

def plotar_grafico(pontos_x, pontos_y, coeficientes):
    # Gera a visualização 2D da interpolação usando Matplotlib.
    margem = (max(pontos_x) - min(pontos_x)) * 0.2 # Define margem de visualização
    if margem == 0: margem = 1.0 # Previne erro se houver apenas 1 ponto

    x_min = min(pontos_x) - margem
    x_max = max(pontos_x) + margem

    # Gera 200 pontos entre min e max
    x_continuo = np.linspace(x_min, x_max, 200)
    y_continuo = [calcular_y_polinomio(x, coeficientes) for x in x_continuo]

    # 2. Plotar
    plt.figure(figsize=(10, 6))

    # Linha do Polinômio
    plt.plot(x_continuo, y_continuo, label='Polinômio Interpolador', color='blue', linewidth=2)

    # Pontos Originais (Dispersão)
    plt.scatter(pontos_x, pontos_y, color='red', zorder=5, s=100, label='Pontos Fornecidos')

    # Detalhes visuais
    plt.title(f"Interpolação Polinomial (Grau {len(pontos_x) - 1})", fontsize=14)
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Linha do eixo 0
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')  # Linha do eixo 0
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Exibir a equação no gráfico
    eq_texto = formatar_equacao_latex(coeficientes)
    plt.text(0.05, 0.95, eq_texto, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    print("\nGerando gráfico...")
    plt.show()

# 2. Lógica da matriz quadratica 3D

def montar_matriz_quadratica_3d(px, py):
    """
    Monta a matriz para ajustar uma superfície quadrática z = f(x,y).
    Modelo: z = c0 + c1*x + c2*y + c3*x^2 + c4*y^2 + c5*x*y
    Necessita idealmente de 6 pontos para uma solução exata quadrada.
    """
    matriz = []
    for x, y in zip(px, py):
        linha = [
            1,
            x,
            y,
            x**2,
            y**2,
            x*y
        ]
        matriz.append(linha)
    return matriz


def formatar_equacao_3d(c):
    # Formata a string da equação da superfície
    return (f"f(x, y) = {c[0]:.2f} + {c[1]:.2f}x + {c[2]:.2f}y + "
            f"{c[3]:.2f}x² + {c[4]:.2f}y² + {c[5]:.2f}xy")


def plotar_superficie(px, py, pz, coef):
    # Gera a visualização 3D da superfície e dos pontos.
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Cria uma malha de coordenadas para desenhar a superfície
    X = np.linspace(min(px)-1, max(px)+1, 40)
    Y = np.linspace(min(py)-1, max(py)+1, 40)

    # Calcula Z para cada ponto da malha usando os coeficientes encontrados
    X, Y = np.meshgrid(X, Y)
    Z = (coef[0] +
     coef[1] * X +
     coef[2] * Y +
     coef[3] * X**2 +
     coef[4] * Y**2 +
     coef[5] * X * Y)

    # Plota a superfície e os pontos originais em 3D
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
    ax.scatter(px, py, pz, color="red", s=50)

    ax.set_title("Superfície Quadrática Interpolada")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


# 3. MAIN 

def main():
    print("===  Curvas e superfícies por pontos especificados: SISTEMA DE INTERPOLAÇÃO ===")
    print("1 - Curva 2D")
    print("2 - Superfície 3D")

    escolha = input("Escolha: ")

    if escolha == "1": # Lógica 2D 
        pontos_x = []
        pontos_y = []

        q = int(input("Quantidade de pontos: "))
        for i in range(q):
            print(f"Ponto {i+1}")
            x = float(input("X: "))
            y = float(input("Y: "))
            pontos_x.append(x)
            pontos_y.append(y)

        A = montar_matriz_vandermonde(pontos_x) # Monta o sistema linear Ac = y
        coef = resolver_sistema_gauss(A, pontos_y) # Resolve para encontrar os coeficientes 'c'

        if coef is None:
            print("Pontos degenerados.")
            return

        plotar_grafico(pontos_x, pontos_y, coef)

    elif escolha == "2": # Lógica 3D
        px = []
        py = []
        pz = []

        q = int(input("Quantidade de pontos (mínimo 6): "))
        for i in range(q):
            print(f"Ponto {i+1}")
            x = float(input("X: "))
            y = float(input("Y: "))
            z = float(input("Z: "))
            px.append(x)
            py.append(y)
            pz.append(z)

        # Monta matriz e resolve
        A = montar_matriz_quadratica_3d(px, py)
        coef = resolver_sistema_gauss(A, pz) # Tenta resolver sistema exato

        if coef is None:
            print("Pontos degenerados.")
            return

        print("\nEquação encontrada:")
        print(formatar_equacao_3d(coef))

        plotar_superficie(px, py, pz, coef)

    else:
        print("Opção inválida.")


# Executa o main
main()
