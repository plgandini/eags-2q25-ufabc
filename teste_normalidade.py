import pandas as pd
from scipy.stats import shapiro

# Carregar os dados
df = pd.read_csv("/home/ubuntu/dados_financeiros.csv")

print("\n--- Teste de Normalidade para Variáveis Quantitativas (Exercício 7) ---\n")

# Variáveis quantitativas para teste de normalidade
variaveis_quantitativas = ["preco_atual", "market_cap", "volatilidade"]

for var in variaveis_quantitativas:
    print(f"\nVariável: {var}\n")
    
    # Remover valores ausentes para a variável atual
    data = df[var].dropna()
    
    if len(data) < 3: # Shapiro-Wilk requires at least 3 data points
        print("   Número insuficiente de dados para realizar o teste de Shapiro-Wilk.")
        continue

    # Hipóteses
    print("   **Hipótese Nula (H0):** A distribuição dos dados da variável {var} é normal.\n")
    print("   **Hipótese Alternativa (H1):** A distribuição dos dados da variável {var} não é normal.\n")

    # Aplicar o teste de Shapiro-Wilk
    stat, p = shapiro(data)

    print(f"   Estatística de Shapiro-Wilk: {stat:.4f}")
    print(f"   P-valor: {p:.4f}\n")

    # Interpretação
    alpha = 0.05
    print(f"   Considerando um nível de significância de {alpha*100}%:\n")
    if p > alpha:
        print("   - **Decisão:** Não rejeitamos a hipótese nula (H0).\n")
        print(f"   - **Conclusão:** Há evidências para afirmar que a distribuição dos dados da variável {var} é normal (ou não há evidências suficientes para rejeitar a normalidade).\n")
    else:
        print("   - **Decisão:** Rejeitamos a hipótese nula (H0).\n")
        print(f"   - **Conclusão:** Não há evidências para afirmar que a distribuição dos dados da variável {var} é normal (ou há evidências suficientes para rejeitar a normalidade).\n")

print("--- Teste de Normalidade Concluído ---\n")


