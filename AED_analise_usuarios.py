import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import csv

# Define o caminho do arquivo pré-processado
caminho_arquivo = "/home/israel/vscode/DoE - Atividade 1/dados_pre_processados_outubro_2018.csv"

# Verifica se o arquivo existe
if not os.path.exists(caminho_arquivo):
    print(f"Erro: Arquivo '{caminho_arquivo}' não encontrado. Reexecute o script de pré-processamento.")
    exit()

# Lê o arquivo inteiro, incluindo media_owner_username
try:
    dados_usuarios = pd.read_csv(
        caminho_arquivo,
        usecols=['media_owner_id', 'created_time', 'created_time_comment', 'comment_id', 'short_code', 'media_owner_username'],
        encoding='utf-8',
        quoting=csv.QUOTE_ALL,
        on_bad_lines='skip',
        low_memory=False,
        escapechar='\\',
        doublequote=True
    )
    print(f"Arquivo lido: {len(dados_usuarios)} linhas")
    print(f"Valores nulos - media_owner_id: {dados_usuarios['media_owner_id'].isna().sum()}")
    print(f"Valores nulos - short_code: {dados_usuarios['short_code'].isna().sum()}")
    print(f"Valores nulos - comment_id: {dados_usuarios['comment_id'].isna().sum()}")
    print(f"Valores nulos - media_owner_username: {dados_usuarios['media_owner_username'].isna().sum()}")
except Exception as e:
    print(f"Erro ao processar o arquivo: {str(e)}")
    exit()

# Verifica se há dados suficientes
if dados_usuarios.empty:
    print("Erro: Subconjunto não contém dados suficientes. Verifique os dados brutos.")
    exit()

# Converte colunas de data
dados_usuarios['created_time'] = pd.to_datetime(dados_usuarios['created_time'], errors='coerce')
dados_usuarios['created_time_comment'] = pd.to_datetime(dados_usuarios['created_time_comment'], errors='coerce')

# 1. Contagem de publicações (short_code únicos por media_owner_id)
publicacoes = dados_usuarios.drop_duplicates(subset=['media_owner_id', 'short_code'])
publicacoes_por_usuario = publicacoes.groupby('media_owner_id')['short_code'].count().sort_values(ascending=False)

# 2. Contagem de comentários recebidos (comment_id únicos por media_owner_id)
dados_comentarios = dados_usuarios[~dados_usuarios['comment_id'].isna()].drop_duplicates(subset=['media_owner_id', 'comment_id'])
comentarios_por_usuario = dados_comentarios.groupby('media_owner_id').size().sort_values(ascending=False)

# 3. Mapeia media_owner_id para media_owner_username
username_map = dados_usuarios.drop_duplicates(subset=['media_owner_id'])[['media_owner_id', 'media_owner_username']].set_index('media_owner_id')['media_owner_username']

# Top 10 com usernames
top_10_publicacoes = publicacoes_por_usuario.head(10).reset_index()
top_10_publicacoes['media_owner_username'] = top_10_publicacoes['media_owner_id'].map(username_map)

top_10_comentarios = comentarios_por_usuario.head(10).reset_index()
top_10_comentarios['media_owner_username'] = top_10_comentarios['media_owner_id'].map(username_map)

# 4. Estatísticas descritivas para publicações
stats_publicacoes = {
    'Média': publicacoes_por_usuario.mean() if not publicacoes_por_usuario.empty else 0,
    'Mediana': publicacoes_por_usuario.median() if not publicacoes_por_usuario.empty else 0,
    'Moda': stats.mode(publicacoes_por_usuario, keepdims=True)[0][0] if not publicacoes_por_usuario.empty else 0,
    'Variância': publicacoes_por_usuario.var(ddof=1) if len(publicacoes_por_usuario) > 1 else 0,
    'Desvio Padrão': publicacoes_por_usuario.std(ddof=1) if len(publicacoes_por_usuario) > 1 else 0,
    'Amplitude': publicacoes_por_usuario.max() - publicacoes_por_usuario.min() if not publicacoes_por_usuario.empty else 0
}

print("\nEstatísticas de Publicações por Usuário (7 de Outubro de 2018):")
for key, value in stats_publicacoes.items():
    print(f"{key}: {value:.2f}")

# 5. Estatísticas descritivas para comentários recebidos
if not comentarios_por_usuario.empty:
    stats_comentarios = {
        'Média': comentarios_por_usuario.mean(),
        'Mediana': comentarios_por_usuario.median(),
        'Moda': stats.mode(comentarios_por_usuario, keepdims=True)[0][0],
        'Variância': comentarios_por_usuario.var(ddof=1) if len(comentarios_por_usuario) > 1 else 0,
        'Desvio Padrão': comentarios_por_usuario.std(ddof=1) if len(comentarios_por_usuario) > 1 else 0,
        'Amplitude': comentarios_por_usuario.max() - comentarios_por_usuario.min()
    }
    print("\nEstatísticas de Comentários Recebidos por Usuário (7 de Outubro de 2018):")
    for key, value in stats_comentarios.items():
        print(f"{key}: {value:.2f}")

# 6. Visualizações
# Gráfico de Barras - Usuários mais ativos (publicações)
plt.figure(figsize=(12, 9))  # Proporção 4:3
plt.bar(top_10_publicacoes['media_owner_username'], top_10_publicacoes['short_code'].count(), color='blue')  # Correção: usar contagem direta
plt.title('Top 10 Usuários - Publicações (7 de Outubro de 2018)')
plt.xlabel('Nome do Usuário')
plt.ylabel('Número de Publicações')
plt.xticks(rotation=45)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.3, top=0.9)  # Ajusta margens
plt.savefig('top_usuarios_publicacoes.png')
plt.close()

# Gráfico de Barras - Usuários mais ativos (comentários recebidos)
plt.figure(figsize=(12, 9))  # Proporção 4:3
plt.bar(top_10_comentarios['media_owner_username'], top_10_comentarios['media_owner_id'].map(comentarios_por_usuario), color='purple')
plt.title('Top 10 Usuários - Comentários Recebidos (7 de Outubro de 2018)')
plt.xlabel('Nome do Usuário')
plt.ylabel('Número de Comentários Recebidos')
plt.xticks(rotation=45)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.3, top=0.9)  # Ajusta margens
plt.savefig('top_usuarios_comentarios.png')
plt.close()

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
\fancyhead[C]{Análise de Usuários - Instagram 7 de Outubro 2018}
\fancyfoot[C]{\thepage}

\titleformat{\section}{\normalfont\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\normalfont\large\bfseries}{\thesubsection}{1em}{}

\begin{document}

\begin{titlepage}
    \centering
    \vspace*{2cm}
    {\Huge\bfseries Análise de Usuários no Instagram\\7 de Outubro 2018\par}
    \vspace{1cm}
    {\Large Israel - DoE Atividade 1\par}
    \vspace{2cm}
    {\large \today\par}
    \vspace{2cm}
\end{titlepage}

\section{Usuários Mais Ativos - Publicações}
\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{top_usuarios_publicacoes.png}
    \caption{Top 10 usuários com mais publicações no dia 7 de outubro de 2018.}
\end{figure}
\clearpage

\section{Usuários Mais Ativos - Comentários Recebidos}
\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{top_usuarios_comentarios.png}
    \caption{Top 10 usuários com mais comentários recebidos no dia 7 de outubro de 2018.}
\end{figure}
\clearpage

\end{document}
"""

# Salva o documento LaTeX sem compilar
with open("analise_usuarios.tex", "w") as f:
    f.write(latex_content)

print("Documento LaTeX gerado: analise_usuarios.tex")
print("Compile manualmente com pdflatex ou online (e.g., Overleaf) para gerar o PDF.")