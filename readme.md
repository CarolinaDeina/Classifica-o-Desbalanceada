#-------------------------------------------------------
# INTRODUÇÃO - DESCRIÇÃO DO PROJETO
#-------------------------------------------------------

Esse código foi desenvolvido para lidar com problemas de classificação binária (duas classes). 
Esse pepiline cobre as etapas de normalização dos dados, balanceamento de classes, cross-validation, seleção das features, previsão das classes utilizando modelos de machine learning (KNN/SVM/RANDOM FOREST), e validação do modelo considerando diferentes métricas de avaliação (Accuracy, sensitivity, specificity, f1-score and auc).
Observe que, antes de utilizar esse código, é necessário realizar etapas de pré-processamentos de dados, como por exemplo, limpeza dos dados (missing data, noise elimination, tratamento de outliers), transformação dos dados (codificação de dados categóricos (one-hot-encoding)), criação de novas variáveis se for o caso, e transformar a variável a ser prevista em 0 e 1 para cada classe a ser prevista.
A base de dados que utilizei para testar a funcionalidade do código já está pré-processada e é uma base de dados de acesso privado.


#-------------------------------------------------------
#  2 - PLOTANDO DAS CLASSES PARA EVIDENCIAR AS PROPORÇÕES
#-------------------------------------------------------

Esta função foi desenvolvida para gerar um gráfico que evidencia as proporções, em percentagem (%), do desbalanceamento entre as classes. Isso permite identificar se é necessário aplicar a função "balanceamento", que é apresentada posteriormente neste código.

Exemplo de Utilização:
Suponha que você tenha um DataFrame denominado "df" com uma coluna para a variável de resposta chamada "CLASSES". Para utilizar a função, siga este exemplo:

plot_countplot_with_proportions(df, "CLASSES")


#-------------------------------------------------------
# 3 - LEITURA DOS DADOS E ALGUNS PARÂMETROS DO CÓDIGO
#-------------------------------------------------------

Nessa etapa é realizada a leitura dos dados e definido alguns parâmetros do código

df = pd.read_csv(" fornecer aqui o caminho da base de dados utilizada ")
target_column = Define qual é a coluna do dataset que contém a variável de resposta (a ser prevista)
scaling_method = define a técnica de escalonamento que será utilizada, podendo ser: 'min_max' ou 'z_score'.
proportionclass = Determina a proporção (%) de desbalanceamento entre as classes que é aceitável para que não se faça resampling nos dados.
imbalanced_classes = plot_classes_with_proportions(df, target_column) ----- com base em uma função de gráfico criada, caso os dados estejam desbalanceados esse parâmetro retornará o valor True, que permitirá que o código faça o balanceamento dos dados, caso contrário (False), não será realizado o balanceamento dos dados. 
res_technique = aqui é definida a técnica para realizar o balanceamento dos dados, podendo ser 'RandomUnderSampler', 'InstanceHardnessThreshold', 'RandomOverSampler' ou 'SMOTE'.


#-------------------------------------------------------
# 4 - DIVISÃO DOS DADOS EM X E Y
#-------------------------------------------------------

Função que separa as variáveis independentes (x1, x2,..., xn) da variável de resposta (y)

Exemplo de uso:
Supondo que você tenha um DataFrame chamado 'df' com a coluna alvo chamada 'column'. Para utilizar a função, siga este exemplo:
df_x, df_y = split_df_into_target_and_features(df, target_column)


#---------------------------------------------------------
# 5 - Escalonamento: NORMALIZAÇÃO ou PADRONIZAÇÃO
#---------------------------------------------------------

O escalonamento é feito para assegurar que todos os dados estejam na mesma escala, facilitando a comparação entre diferentes variáveis.
Na normalização ('min_max') os valores são ajustados para um intervalo específico, como 0 e 1.
A padronização ('z_score') é o processo de transformar os dados de forma que eles tenham uma média de 0 e um desvio padrão de 1.

Na função abaixo, você pode passar o parâmetro method para escolher entre 'min_max' e 'z_score' para indicar o tipo de escalonamento desejado. 
Se um método não reconhecido for fornecido, a função lançará um erro. 

Exemplo de uso da função:
Supondo que você tenha um dataframe "x" que contenha todas as suas variáveis independentes e escolha utilizar o método 'min_max' que cria uma escala entre 0 e 1 em todos os seus dados. Você pode chamar a função assim:

x_normalized = normaliz(x, 'min_max')

Caso queira fazer o processo de padronização dos dados: 

x_normalized = normaliz(x, 'z_score')


#---------------------------------------------------------
# 6 - METRICAS DE AVALIAÇÃO DOS DADOS
#---------------------------------------------------------


Por se tratar de um problema de classificação, diferentes métricas podem ser utilizadas para avaliar os resultados da classificação:

*Acurácia: Mede a proporção de previsões corretas em relação ao total de previsões.

*Precisão: Mede a proporção de verdadeiros positivos em relação ao total de previsões positivas.

*Recall ou sensibilidade: Mede a proporção de verdadeiros positivos em relação ao total de casos positivos reais.

*F1-Score: Combina precisão e recall em uma única medida que equilibra o trade-off entre eles, através de uma média harmônica.

*Área sob a Curva ROC (AUC): Avalia a capacidade do modelo de distinguir entre classes positivas e negativas em um problema de classificação binária. Quanto maior, melhor a capacidade de discriminação do modelo.

Exemplo de Utilização da Função:
Suponha que os dados de entrada sejam os dados reais de saída esperada (y_true) e os dados previstos pelo algoritmo (y_pred). Para chamar a função, proceda da seguinte maneira:

acuracia, precisao, recall, f1_score, auc = classification_metrics(y_true, y_pred)


#---------------------------------------------------------
# 7 - BALANCEAMENTO DOS DADOS
#---------------------------------------------------------

O balanceamento de dados é uma etapa crucial no pré-processamento de dados, onde desequilíbrios nas classes podem levar a resultados enviesados ou modelos com desempenho inadequado. 
Existem várias abordagens para o balanceamento de dados no nível dos dados:

RandomUnderSampler: seleciona aleatoriamente um subconjunto das instâncias da classe majoritária (para ser descartadas) para igualar o número de instâncias da classe minoritária.

InstanceHardnessThreshold: usa uma medida de "dureza" para determinar quais instâncias são mais difíceis de classificar corretamente. Ele remove as instâncias mais "fáceis" da classe majoritária, com base nessa medida, para tentar equilibrar as classes.

RandomOverSampler: Ao contrário do RandomUnderSampler, este método replica aleatoriamente instâncias da classe minoritária até que o número de instâncias de ambas as classes seja equilibrado.

SMOTE (Synthetic Minority Over-sampling Technique): gera instâncias sintéticas da classe minoritária, criando novos exemplos interpolados entre instâncias existentes. 


#---------------------------------------------------------
# 8 - BALANCEAMENTO DOS DADOS, se 'imbalanced_classes' for 'true'
#---------------------------------------------------------

A função de Balanceamento dos dados criada anteriormente, só é aplicada se a condição de desbalanceamento nos dados for verdadeira.