# PP2.1 - Perceptron

Implementação orientada a objetos do algoritmo Perceptron para classificação binária.

## Equipe e Identificador

- **Ana Klissia Furtado Martins** — Matrícula: `2415310025`
- **Nathalia Nóbrega Silva** — Matrícula: `2415310021`

**Identificador do Grupo:** `2`  
*(Cálculo: `(5 + 1) % 4 = 2`, associando o arquivo `data220262.txt` para a Parte II)*

---

## Estrutura de Arquivos

```text
PP2.1_Perceptron/
├── PP21_Perceptron.ipynb   # Notebook principal com a execução das Partes I, II e III
├── dataAll20262.txt        # Base de dados da Parte I
├── data220262.txt          # Base de dados da Parte II (identificador 2)
├── dataHoldout20262.txt    # Base de dados de teste da Parte III (holdout)
├── perceptron/             # Pacote orientado a objetos
│   ├── __init__.py         # Exportações do pacote
│   ├── projeto.py          # Definição do grupo, cálculo do id e caminhos
│   ├── dados.py            # Carregamento, partição e normalização
│   ├── modelo.py           # Algoritmo Perceptron e histórico de treinamento
│   ├── avaliacao.py        # Métricas de desempenho (acurácia, F1, etc.)
│   ├── experimento.py      # Grid de hiperparâmetros e 30 repetições
│   ├── formatacao.py       # Formatação de tabelas (PrettyTable)
│   └── visualizacao.py     # Gráficos de dispersão, fronteiras e épocas
└── README.md
```

---

## Como Executar

### 1. Instalar dependências

```bash
pip install numpy matplotlib scikit-learn prettytable notebook
```

### 2. Executar o notebook

Abra e execute o notebook no Jupyter Notebook, VS Code ou Google Colab:

```bash
jupyter notebook PP21_Perceptron.ipynb
```

Execute as células em sequência para reproduzir os experimentos e visualizações das Partes I, II e III.
