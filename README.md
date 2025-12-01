# Curvas e Superfícies por Pontos Especificados

### Interpolação 2D e 3D usando Python

Este projeto implementa um sistema interativo para gerar curvas e superfícies a partir de pontos fornecidos pelo usuário.
Calculando a função polinomial (2D) ou a superfície quadrática (3D) que passa pelos pontos especificados, utilizando **Eliminação de Gauss**.

Além disso, o programa exibe gráficos com Matplotlib!

## Funcionalidades

### Interpolação 2D

- Entrada de **n pontos** no plano
- Construção automática da **matriz de Vandermonde**
- Cálculo do **polinômio interpolador**
- Plotagem contendo:
    - Curva interpolada
    - Pontos fornecidos
    - Equação formatada dentro do gráfico

---

### Interpolação de Superfície 3D

- Entrada de **pelo menos 6 pontos** no espaço 3D
- Ajuste da superfície quadrática:

f(x,y)=a0+a1x+a2y+a3x2+c4y2+c5xy

- Resolução do sistema linear correspondente
- Plotagem 3D completa da superfície + pontos de entrada

---

## Como executar

### 1. Instalar dependências

Certifique-se de que o Python 3.x está instalado.

Instale as bibliotecas:

```bash
pip install matplotlib numpy

```

### 2. Executar o código

```bash
python curvas_e_superficies.py

```

### 3. Escolher a opção no menu

- **1** → Interpolação de curva 2D
- **2** → Superfície 3D

Insira os pontos quando solicitado.
