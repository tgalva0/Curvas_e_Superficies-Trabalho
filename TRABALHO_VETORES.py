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

def plotar_grafico(pontos_x, pontos_y, coeficientes):

    margem = (max(pontos_x) - min(pontos_x)) * 0.2
    if margem == 0: margem = 1.0

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

# 2. PARTE DO 3D

def montar_matriz_quadratica_3d(px, py):
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
    return (f"f(x, y) = {c[0]:.2f} + {c[1]:.2f}x + {c[2]:.2f}y + "
            f"{c[3]:.2f}x² + {c[4]:.2f}y² + {c[5]:.2f}xy")


def plotar_superficie(px, py, pz, coef):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    X = np.linspace(min(px)-1, max(px)+1, 40)
    Y = np.linspace(min(py)-1, max(py)+1, 40)

    X, Y = np.meshgrid(X, Y)
    Z = (coef[0] +
     coef[1] * X +
     coef[2] * Y +
     coef[3] * X**2 +
     coef[4] * Y**2 +
     coef[5] * X * Y)


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

    if escolha == "1":
        pontos_x = []
        pontos_y = []

        q = int(input("Quantidade de pontos: "))
        for i in range(q):
            print(f"Ponto {i+1}")
            x = float(input("X: "))
            y = float(input("Y: "))
            pontos_x.append(x)
            pontos_y.append(y)

        A = montar_matriz_vandermonde(pontos_x)
        coef = resolver_sistema_gauss(A, pontos_y)

        if coef is None:
            print("Pontos degenerados.")
            return

        plotar_grafico(pontos_x, pontos_y, coef)

    elif escolha == "2":
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

        A = montar_matriz_quadratica_3d(px, py)
        coef = resolver_sistema_gauss(A, pz)

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
