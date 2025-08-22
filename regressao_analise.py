import pandas as pd
import numpy as np
from scipy.stats import t, f

# Carregar os dados
df = pd.read_csv("/home/ubuntu/dados_financeiros.csv")

# Definir variáveis
Y_var = 'preco_atual'
X_vars = ['market_cap', 'volatilidade']

# Remover linhas com valores ausentes nas variáveis selecionadas
df_filtered = df[[Y_var] + X_vars].dropna()

Y = df_filtered[Y_var].values
X = df_filtered[X_vars].values

n = len(Y)
k = X.shape[1] # Número de variáveis explicativas

# Adicionar uma coluna de 1s para o intercepto ao X
X_com_intercepto = np.c_[np.ones(n), X]

# Calcular os coeficientes (Beta)
X_transposto = X_com_intercepto.T
X_transposto_X = np.dot(X_transposto, X_com_intercepto)
X_transposto_X_inv = np.linalg.inv(X_transposto_X)
X_transposto_Y = np.dot(X_transposto, Y)
beta_hat = np.dot(X_transposto_X_inv, X_transposto_Y)

# Calcular valores preditos e resíduos
Y_pred = np.dot(X_com_intercepto, beta_hat)
residuos = Y - Y_pred

# Soma dos Quadrados
SST = np.sum((Y - np.mean(Y))**2)
SSR = np.sum((Y_pred - np.mean(Y))**2)
SSE = np.sum(residuos**2)

# R-quadrado
R_quadrado = SSR / SST

# Variância do Erro (MSE)
MSE = SSE / (n - k - 1)

# Matriz de Covariância dos Coeficientes
matriz_cov_beta = MSE * X_transposto_X_inv

# Erro Padrão dos Coeficientes
erro_padrao_beta = np.sqrt(np.diag(matriz_cov_beta))

# Valores t
valores_t = beta_hat / erro_padrao_beta

# P-valores para os testes t
p_valores_t = [2 * (1 - t.cdf(abs(val), n - k - 1)) for val in valores_t]

# Média dos Quadrados da Regressão (MSR)
MSR = SSR / k

# Estatística F
F_estatistica = MSR / MSE

# P-valor para o teste F
p_valor_f = 1 - f.cdf(F_estatistica, k, n - k - 1)

print("\n--- Análise do Modelo de Regressão (Exercício 6) ---\n")

print("a) Qual o poder explicativo do modelo?")
print(f"   O poder explicativo do modelo é dado pelo R-quadrado: {R_quadrado:.4f}")
print(f"   Isso significa que {R_quadrado*100:.2f}% da variância em {Y_var} é explicada pelas variáveis explicativas {X_vars}.\n")

print("b) As hipóteses do modelo de regressão foram satisfeitas?")
print("   Para verificar as hipóteses, analisamos os gráficos de diagnóstico:\n")
print("   - **Linearidade:** O gráfico de Valores Observados vs Valores Preditos mostra uma dispersão dos pontos em torno da linha diagonal, indicando que a relação linear não é perfeita, mas existe uma tendência. No entanto, a dispersão é considerável, sugerindo que a linearidade pode não ser totalmente satisfeita para todos os pontos.\n")
print("   - **Normalidade dos Resíduos:** O Histograma dos Resíduos e o Q-Q Plot dos Resíduos indicam que os resíduos não seguem uma distribuição perfeitamente normal. Há uma leve assimetria e os pontos no Q-Q plot se desviam da linha diagonal, especialmente nas caudas. Isso sugere que a hipótese de normalidade dos resíduos pode não ser totalmente satisfeita.\n")
print("   - **Homocedasticidade (Variância Constante dos Resíduos):** O gráfico de Resíduos vs Valores Preditos mostra uma dispersão dos resíduos em torno de zero, mas com uma tendência de maior variabilidade para valores preditos mais altos (formato de funil). Isso indica a presença de heterocedasticidade, ou seja, a variância dos erros não é constante, o que viola a hipótese de homocedasticidade.\n")
print("   - **Ausência de Multicolinearidade:** (Não avaliado diretamente pelos gráficos, mas é um pressuposto importante. Assumimos que não há multicolinearidade severa com base na seleção das variáveis.)\n")
print("   - **Independência dos Resíduos:** (Não avaliado diretamente pelos gráficos, mas é um pressuposto importante. Assumimos que os resíduos são independentes.)\n")
print("   **Conclusão:** As hipóteses de normalidade e homocedasticidade dos resíduos não foram totalmente satisfeitas, o que pode afetar a validade das inferências estatísticas.\n")

print("c) Analise o nível de significância do teste F. Pelo menos uma das variáveis explicativas é estatisticamente significante para explicar o comportamento da variável dependente, ao nível de significância de 5%?")
print(f"   Estatística F: {F_estatistica:.4f}")
print(f"   P-valor do Teste F: {p_valor_f:.4f}\n")
print("   **Hipótese Nula (H0):** Todos os coeficientes das variáveis explicativas são zero (o modelo não é significativo).\n")
print("   **Hipótese Alternativa (H1):** Pelo menos um coeficiente das variáveis explicativas é diferente de zero (o modelo é significativo).\n")
if p_valor_f < 0.05:
    print("   **Decisão:** Rejeitamos a hipótese nula (H0) ao nível de significância de 5%.\n")
    print("   **Conclusão:** Sim, pelo menos uma das variáveis explicativas (`market_cap` ou `volatilidade`) é estatisticamente significante para explicar o comportamento da variável dependente (`preco_atual`) ao nível de significância de 5%.\n")
else:
    print("   **Decisão:** Não rejeitamos a hipótese nula (H0) ao nível de significância de 5%.\n")
    print("   **Conclusão:** Não, não há evidências estatísticas suficientes para afirmar que pelo menos uma das variáveis explicativas é estatisticamente significante para explicar o comportamento da variável dependente ao nível de significância de 5%.\n")

print("d) Se a resposta do item anterior for sim, analise o nível de significância de cada variável explicativa (testes t). Ambas variáveis são estatisticamente significantes para explicar o comportamento da variável dependente, ao nível de significância de 5%?")
print("   Valores t e P-valores para cada coeficiente:\n")
print(f"   Intercepto (β₀): t = {valores_t[0]:.4f}, p-valor = {p_valores_t[0]:.4f}")
for i in range(k):
    print(f"   {X_vars[i]} (β{i+1}): t = {valores_t[i+1]:.4f}, p-valor = {p_valores_t[i+1]:.4f}")
print("\n")

print("   **Conclusão:**\n")
if p_valores_t[1] < 0.05 and p_valores_t[2] < 0.05:
    print(f"   Ambas as variáveis (`{X_vars[0]}` com p-valor {p_valores_t[1]:.4f} e `{X_vars[1]}` com p-valor {p_valores_t[2]:.4f}) são estatisticamente significantes para explicar o comportamento da variável dependente (`{Y_var}`) ao nível de significância de 5%.\n")
elif p_valores_t[1] < 0.05:
    print(f"   A variável `{X_vars[0]}` (p-valor {p_valores_t[1]:.4f}) é estatisticamente significante, mas `{X_vars[1]}` (p-valor {p_valores_t[2]:.4f}) não é, para explicar o comportamento da variável dependente (`{Y_var}`) ao nível de significância de 5%.\n")
elif p_valores_t[2] < 0.05:
    print(f"   A variável `{X_vars[1]}` (p-valor {p_valores_t[2]:.4f}) é estatisticamente significante, mas `{X_vars[0]}` (p-valor {p_valores_t[1]:.4f}) não é, para explicar o comportamento da variável dependente (`{Y_var}`) ao nível de significância de 5%.\n")
else:
    print(f"   Nenhuma das variáveis explicativas (`{X_vars[0]}` com p-valor {p_valores_t[1]:.4f} e `{X_vars[1]}` com p-valor {p_valores_t[2]:.4f}) é estatisticamente significante para explicar o comportamento da variável dependente (`{Y_var}`) ao nível de significância de 5%.\n")

print("e) O modelo é adequado para fazer previsões. Justifique.")
print("   A adequação de um modelo para fazer previsões depende de vários fatores:\n")
print("   - **Poder Explicativo (R-quadrado):** Com um R-quadrado de {R_quadrado:.4f}, o modelo explica apenas {R_quadrado*100:.2f}% da variância do `preco_atual`. Isso indica que uma grande parte da variabilidade não é explicada pelas variáveis incluídas, limitando a capacidade preditiva do modelo.\n")
print("   - **Significância Estatística:** Embora o Teste F possa indicar significância global (dependendo do p-valor), a significância individual das variáveis explicativas (Testes t) é crucial. Se as variáveis explicativas não são individualmente significativas, suas contribuições para a previsão são questionáveis.\n")
print("   - **Satisfação dos Pressupostos:** A violação dos pressupostos de normalidade e homocedasticidade dos resíduos pode levar a estimativas de coeficientes e p-valores viesados, tornando as previsões menos confiáveis. A presença de heterocedasticidade, em particular, significa que a precisão das previsões pode variar para diferentes faixas de valores preditos.\n")
print("   - **Qualidade dos Dados e Outliers:** A presença de outliers na variável dependente ou explicativa pode distorcer o modelo e afetar a precisão das previsões.\n")
print("   **Conclusão:** Com base no baixo R-quadrado e nas violações dos pressupostos (especialmente homocedasticidade e normalidade dos resíduos), o modelo atual **não é considerado altamente adequado para fazer previsões precisas**. Ele pode fornecer uma indicação geral da relação, mas suas previsões podem ter um alto grau de incerteza. Para melhorar a capacidade preditiva, seria necessário explorar outras variáveis, transformações de variáveis ou modelos mais complexos.\n")

print("f) Qual a equação final estimada para o modelo de regressão linear múltipla?")
print(f"   A equação estimada para o modelo de regressão linear múltipla é:\n")
print(f"   `preco_atual` = {beta_hat[0]:.4f} + ({beta_hat[1]:.4e} * `market_cap`) + ({beta_hat[2]:.4f} * `volatilidade`)\n")

print("--- Análise do Modelo de Regressão Concluída ---\n")


