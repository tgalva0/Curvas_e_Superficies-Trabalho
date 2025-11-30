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

# 3D - PLANO: z = a + b*x + c*y
def montar_matriz_plano_3d(pontos_x, pontos_y):
    matriz = []
    for x, y in zip(pontos_x, pontos_y):
        matriz.append([1, x, y])
    return matriz

def calcular_z_plano(x, y, coef):
    return coef[0] + coef[1]*x + coef[2]*y

def formatar_equacao_plano(coef):
    return f"z = {coef[0]:.2f} + {coef[1]:.2f}*x + {coef[2]:.2f}*y"

def plotar_plano_3d(pontos_x, pontos_y, pontos_z, coef):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    X_grid, Y_grid = np.meshgrid(
        np.linspace(min(pontos_x)-1, max(pontos_x)+1, 20),
        np.linspace(min(pontos_y)-1, max(pontos_y)+1, 20)
    )
    Z_grid = coef[0] + coef[1]*X_grid + coef[2]*Y_grid
    ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.5, color='cyan')
    ax.scatter(pontos_x, pontos_y, pontos_z, color='red', s=100)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Plano 3D que passa pelos 3 pontos")
    plt.show()

# MAIN
def main():
    print("=== SISTEMA DE INTERPOLAÇÃO POLINOMIAL ===")
    print("Escolha o tipo de interpolação:")
    print("1 - 2D (curva)")
    print("2 - 3D (plano)")

    while True:
        opcao = input("Digite 1 para 2D ou 2 para 3D: ")
        if opcao in ['1','2']: break
        print("Opção inválida. Digite 1 ou 2.")

    if opcao == '1':
        pontos_x = []
        pontos_y = []
        while True:
            try:
                qtd = int(input("\nQuantos pontos deseja inserir? (Min: 2): "))
                if qtd >= 2: break
                print("Precisa de pelo menos 2 pontos.")
            except ValueError:
                print("Digite um número inteiro.")

        for i in range(qtd):
            print(f"\n--- Ponto {i + 1} ---")
            while True:
                try:
                    px = float(input("Coordenada X: "))
                    py = float(input("Coordenada Y: "))
                    if px in pontos_x:
                        print("Erro: X já existe. Para ser função, X deve ser único.")
                        continue
                    pontos_x.append(px)
                    pontos_y.append(py)
                    break
                except ValueError:
                    print("Digite apenas números.")

        print("\n[1/3] Montando Matriz de Vandermonde...")
        A = montar_matriz_vandermonde(pontos_x)
        print("[2/3] Resolvendo Sistema Linear (Gauss)...")
        coeficientes = resolver_sistema_gauss(A, pontos_y)
        if coeficientes is None:
            print("ERRO CRÍTICO: Pontos degenerados (Matriz Singular).")
        else:
            print("\n>>> SUCESSO! Coeficientes encontrados:")
            for g, c in enumerate(coeficientes):
                print(f"   Termo x^{g}: {c:.4f}")
            print(f"\nEquação: {formatar_equacao_latex(coeficientes)}")
            print("[3/3] Abrindo visualização gráfica...")
            plotar_grafico(pontos_x, pontos_y, coeficientes)

    else:  # 3D
        pontos_x, pontos_y, pontos_z = [], [], []
        print("\nInsira 3 pontos (não colineares) para formar um plano:")
        for i in range(3):
            print(f"\n--- Ponto {i + 1} ---")
            while True:
                try:
                    px = float(input("X: "))
                    py = float(input("Y: "))
                    pz = float(input("Z: "))
                    pontos_x.append(px)
                    pontos_y.append(py)
                    pontos_z.append(pz)
                    break
                except ValueError:
                    print("Digite apenas números.")

        print("\n[1/3] Montando matriz do plano...")
        A = montar_matriz_plano_3d(pontos_x, pontos_y)
        print("[2/3] Resolvendo sistema linear...")
        coef = resolver_sistema_gauss(A, pontos_z)
        if coef is None:
            print("ERRO CRÍTICO: Pontos degenerados (não é possível formar um plano único).")
        else:
            print("\n>>> SUCESSO! Coeficientes encontrados:")
            print(formatar_equacao_plano(coef))
            print("[3/3] Gerando gráfico 3D...")
            plotar_plano_3d(pontos_x, pontos_y, pontos_z, coef)

if __name__ == "__main__":
    main()
