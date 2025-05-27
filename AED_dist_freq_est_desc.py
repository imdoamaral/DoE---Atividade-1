import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from fitter import Fitter
import os
import csv

# Define o caminho do arquivo
caminho_arquivo = "/home/israel/vscode/DoE - Atividade 1/dados_pre_processados_outubro_2018.csv"

# Verifica se o arquivo existe
if not os.path.exists(caminho_arquivo):
    print(f"Erro: Arquivo '{caminho_arquivo}' não encontrado. Reexecute o script de pré-processamento.")
    exit()

# Define o tamanho do chunk
tamanho_chunk = 10000

# Inicializa séries para contagens por hora
posts_por_hora = pd.Series(dtype=int)
comentarios_por_hora = pd.Series(dtype=int)

# Função para processar cada chunk
def processar_chunk(chunk, chunk_num):
    print(f"Processando chunk {chunk_num}: {len(chunk)} linhas")
    print(f"Valores nulos - created_time: {chunk['created_time'].isna().sum()}, created_time_comment: {chunk['created_time_comment'].isna().sum()}")
    
    # Converte colunas de data
    chunk['created_time'] = pd.to_datetime(chunk['created_time'], errors='coerce')
    chunk['created_time_comment'] = pd.to_datetime(chunk['created_time_comment'], errors='coerce')
    
    # Filtra created_time_comment para 7 de outubro de 2018
    data_alvo = pd.to_datetime('2018-10-07')
    mascara_data_coment = chunk['created_time_comment'].dt.date == data_alvo.date()
    chunk = chunk[mascara_data_coment | chunk['created_time_comment'].isna()]
    
    # Extrai hora
    chunk['hora_post'] = chunk['created_time'].dt.floor('h')
    chunk['hora_comentario'] = chunk['created_time_comment'].dt.floor('h')
    
    # Conta publicações (usando short_code único por hora)
    posts = chunk.drop_duplicates(subset=['short_code']).groupby('hora_post').size()
    comentarios = chunk.groupby('hora_comentario').size() if not chunk['created_time_comment'].isna().all() else pd.Series(dtype=int)
    
    print(f"Publicações por hora: {len(posts)}, Comentários por hora: {len(comentarios)}")
    return posts, comentarios

# Lê os chunks
try:
    chunk_count = 0
    for chunk in pd.read_csv(
        caminho_arquivo,
        chunksize=tamanho_chunk,
        usecols=['created_time', 'created_time_comment', 'short_code'],
        encoding='utf-8',
        quoting=csv.QUOTE_ALL,
        on_bad_lines='skip',
        low_memory=False,
        escapechar='\\',
        doublequote=True
    ):
        chunk_count += 1
        posts, comentarios = processar_chunk(chunk, chunk_count)
        posts_por_hora = posts_por_hora.add(posts, fill_value=0)
        comentarios_por_hora = comentarios_por_hora.add(comentarios, fill_value=0)
except Exception as e:
    print(f"Erro ao processar o subconjunto: {str(e)}")
    exit()

# Remove NaN e converte para inteiros
posts_por_hora = posts_por_hora.dropna().astype(int)
comentarios_por_hora = comentarios_por_hora.dropna().astype(int)

# Verifica se há dados suficientes
if posts_por_hora.empty:
    print("Erro: Subconjunto não contém dados suficientes para publicações. Verifique os dados brutos.")
    exit()

# Estatísticas descritivas para publicações
mean_posts = posts_por_hora.mean()
median_posts = posts_por_hora.median()
mode_posts = stats.mode(posts_por_hora, keepdims=True)[0][0]
variance_posts = posts_por_hora.var(ddof=1) if len(posts_por_hora) > 1 else 0
std_dev_posts = posts_por_hora.std(ddof=1) if len(posts_por_hora) > 1 else 0
range_posts = posts_por_hora.max() - posts_por_hora.min()

# Imprime estatísticas para publicações
print("\nEstatísticas para Publicações por Hora (Subconjunto):")
print(f"Média: {mean_posts:.2f}")
print(f"Mediana: {median_posts:.2f}")
print(f"Moda: {mode_posts:.2f}")
print(f"Variância: {variance_posts:.2f}")
print(f"Desvio Padrão: {std_dev_posts:.2f}")
print(f"Amplitude: {range_posts:.2f}")

# Estatísticas descritivas para comentários (se disponíveis)
if not comentarios_por_hora.empty:
    mean_comments = comentarios_por_hora.mean()
    median_comments = comentarios_por_hora.median()
    mode_comments = stats.mode(comentarios_por_hora, keepdims=True)[0][0]
    variance_comments = comentarios_por_hora.var(ddof=1) if len(comentarios_por_hora) > 1 else 0
    std_dev_comments = comentarios_por_hora.std(ddof=1) if len(comentarios_por_hora) > 1 else 0
    range_comments = comentarios_por_hora.max() - comentarios_por_hora.min()
    
    print("\nEstatísticas para Comentários por Hora (Subconjunto):")
    print(f"Média: {mean_comments:.2f}")
    print(f"Mediana: {median_comments:.2f}")
    print(f"Moda: {mode_comments:.2f}")
    print(f"Variância: {variance_comments:.2f}")
    print(f"Desvio Padrão: {std_dev_comments:.2f}")
    print(f"Amplitude: {range_comments:.2f}")

# Visualizações
plt.figure(figsize=(12, 9))  # Proporção 4:3

# Histograma - Publicações
plt.subplot(2, 2, 1)
plt.hist(posts_por_hora, bins=10, density=True, alpha=0.7, color='blue')
plt.title('Histograma - Publicações por Hora')
plt.xlabel('Número de Publicações')
plt.ylabel('Densidade')

# Box Plot - Publicações
plt.subplot(2, 2, 2)
sns.boxplot(data=posts_por_hora, color='green')
plt.title('Box Plot - Publicações por Hora')
plt.ylabel('Número de Publicações')

# CDF - Publicações
plt.subplot(2, 2, 3)
sorted_posts = np.sort(posts_por_hora)
cdf = np.arange(1, len(sorted_posts) + 1) / len(sorted_posts)
plt.plot(sorted_posts, cdf, color='red')
plt.title('CDF - Publicações por Hora')
plt.xlabel('Número de Publicações')
plt.ylabel('Probabilidade Cumulativa')

# Histograma - Comentários (se disponíveis)
if not comentarios_por_hora.empty:
    plt.subplot(2, 2, 4)
    plt.hist(comentarios_por_hora, bins=10, density=True, alpha=0.7, color='purple')
    plt.title('Histograma - Comentários por Hora')
    plt.xlabel('Número de Comentários')
    plt.ylabel('Densidade')

plt.tight_layout()
plt.savefig('instagram_exploratory_analysis_subset.png')
plt.close()

# Ajuste de distribuição para publicações
f = Fitter(posts_por_hora, distributions=['expon', 'gamma', 'lognorm'])
f.fit()
f.summary()

# Gera o documento LaTeX para o PDF (sem compilar)
latex_content = r"""
\documentclass[a4paper]{article}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{fancyhdr}

\geometry{margin=1in}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[C]{Análise Exploratória - Instagram 7 de Outubro 2018}
\fancyfoot[C]{\thepage}

\titleformat{\section}{\normalfont\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\normalfont\large\bfseries}{\thesubsection}{1em}{}

\begin{document}

\begin{titlepage}
    \centering
    \vspace*{2cm}
    {\Huge\bfseries Análise Exploratória de Publicações e Comentários no Instagram\\7 de Outubro 2018\par}
    \vspace{1cm}
    {\Large Israel - DoE Atividade 1\par}
    \vspace{2cm}
    {\large \today\par}
    \vspace{2cm}
\end{titlepage}

\section{Histograma e Box Plot de Publicações por Hora}
\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{instagram_exploratory_analysis_subset.png}
    \caption{Histograma, Box Plot e CDF de publicações por hora, com histograma de comentários (se disponíveis).}
\end{figure}
\clearpage

\end{document}
"""

# Salva o documento LaTeX sem compilar
with open("analise_exploratoria.tex", "w") as f:
    f.write(latex_content)

print("Documento LaTeX gerado: analise_exploratoria.tex")
print("Compile manualmente com pdflatex ou online (e.g., Overleaf) para gerar o PDF.")