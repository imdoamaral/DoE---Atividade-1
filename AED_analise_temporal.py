import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import subprocess

# Define o caminho do arquivo pré-processado
caminho_arquivo = "/home/israel/vscode/DoE - Atividade 1/dados_pre_processados_outubro_2018.csv"

# Verifica se o arquivo existe
if not os.path.exists(caminho_arquivo):
    print(f"Erro: Arquivo '{caminho_arquivo}' não encontrado. Reexecute o script de pré-processamento.")
    exit()

# Define o tamanho do chunk e máximo de chunks
tamanho_chunk = 5000
max_chunks = 10

# Inicializa séries para contagens por hora
serie_posts = pd.Series(dtype=int)
serie_comentarios = pd.Series(dtype=int)

# Função para processar cada chunk
def processar_chunk_temporal(chunk, chunk_num):
    print(f"Processando chunk {chunk_num}: {len(chunk)} linhas")
    print(f"Valores nulos - created_time: {chunk['created_time'].isna().sum()}, created_time_comment: {chunk['created_time_comment'].isna().sum()}")
    
    # Converte colunas de data
    chunk['created_time'] = pd.to_datetime(chunk['created_time'], errors='coerce')
    chunk['created_time_comment'] = pd.to_datetime(chunk['created_time_comment'], errors='coerce')
    
    # Filtra apenas datas de outubro de 2018
    mascara_oct = (chunk['created_time'].dt.month == 10) & (chunk['created_time'].dt.year == 2018)
    chunk = chunk[mascara_oct]
    if not chunk.empty:
        # Extrai hora completa (data + hora)
        chunk['hora_post'] = chunk['created_time'].dt.floor('h')
        chunk['hora_comentario'] = chunk['created_time_comment'].dt.floor('h')
        
        # Conta publicações e comentários por hora
        posts = chunk.groupby('hora_post').size()
        comentarios = chunk.groupby('hora_comentario').size() if not chunk['created_time_comment'].isna().all() else pd.Series(dtype=int)
        
        print(f"Publicações por hora: {len(posts)}, Comentários por hora: {len(comentarios)}")
        return posts, comentarios
    return pd.Series(dtype=int), pd.Series(dtype=int)

# Lê até max_chunks
try:
    chunk_count = 0
    for chunk in pd.read_csv(
        caminho_arquivo,
        chunksize=tamanho_chunk,
        usecols=['created_time', 'created_time_comment'],
        encoding='utf-8',
        quoting=csv.QUOTE_ALL,
        on_bad_lines='skip',
        low_memory=False,
        escapechar='\\',
        doublequote=True
    ):
        chunk_count += 1
        posts, comentarios = processar_chunk_temporal(chunk, chunk_count)
        serie_posts = serie_posts.add(posts, fill_value=0)
        serie_comentarios = serie_comentarios.add(comentarios, fill_value=0)
        if chunk_count >= max_chunks:
            break
except Exception as e:
    print(f"Erro ao processar o subconjunto: {str(e)}")
    exit()

# Remove NaN e converte para inteiros
serie_posts = serie_posts.dropna().astype(int)
serie_comentarios = serie_comentarios.dropna().astype(int)

# Verifica se há dados suficientes
if serie_posts.empty:
    print("Erro: Subconjunto não contém dados suficientes para publicações. Verifique os dados brutos.")
    exit()

# Reindexa para garantir continuidade, restrito a outubro de 2018
data_inicio = pd.Timestamp('2018-10-01 00:00:00')
data_fim = pd.Timestamp('2018-10-31 23:00:00')
indice_horas = pd.date_range(start=data_inicio, end=data_fim, freq='h')
serie_posts = serie_posts.reindex(indice_horas, fill_value=0)
serie_comentarios = serie_comentarios.reindex(indice_horas, fill_value=0)

# Calcula médias móveis (janela de 24 horas para capturar tendências diárias)
janela_media_movel = 24
media_movel_posts = serie_posts.rolling(window=janela_media_movel, min_periods=1).mean()
media_movel_comentarios = serie_comentarios.rolling(window=janela_media_movel, min_periods=1).mean()

# Identifica períodos de alta e baixa atividade (percentis 90 e 10)
limiar_alta_posts = serie_posts.quantile(0.90)
limiar_baixa_posts = serie_posts.quantile(0.10)
limiar_alta_comentarios = serie_comentarios.quantile(0.90) if not serie_comentarios.empty else 0
limiar_baixa_comentarios = serie_comentarios.quantile(0.10) if not serie_comentarios.empty else 0

print("\nPeríodos de Alta e Baixa Atividade (Publicações):")
print(f"Limiar Alta (90%): {limiar_alta_posts:.2f}")
print(f"Limiar Baixa (10%): {limiar_baixa_posts:.2f}")
if not serie_comentarios.empty:
    print("\nPeríodos de Alta e Baixa Atividade (Comentários):")
    print(f"Limiar Alta (90%): {limiar_alta_comentarios:.2f}")
    print(f"Limiar Baixa (10%): {limiar_baixa_comentarios:.2f}")

# Calcula sazonalidade (média por hora do dia)
serie_posts_sazonal = serie_posts.groupby(serie_posts.index.hour).mean()
serie_comentarios_sazonal = serie_comentarios.groupby(serie_comentarios.index.hour).mean() if not serie_comentarios.empty else pd.Series()

# Gera os gráficos e salva como PNG temporariamente
# Gráfico 1: Séries Temporais de Publicações
plt.figure(figsize=(15, 5))
plt.plot(serie_posts.index, serie_posts, label='Publicações por Hora', color='blue', alpha=0.5)
plt.plot(media_movel_posts.index, media_movel_posts, label=f'Média Móvel ({janela_media_movel}h)', color='red')
plt.axhline(y=limiar_alta_posts, color='green', linestyle='--', label='Limiar Alta Atividade')
plt.axhline(y=limiar_baixa_posts, color='orange', linestyle='--', label='Limiar Baixa Atividade')
plt.title('Série Temporal de Publicações por Hora')
plt.xlabel('Data e Hora')
plt.ylabel('Número de Publicações')
plt.legend()
plt.savefig('serie_temporal_publicacoes.png')
plt.close()

# Gráfico 2: Séries Temporais de Comentários
if not serie_comentarios.empty:
    plt.figure(figsize=(15, 5))
    plt.plot(serie_comentarios.index, serie_comentarios, label='Comentários por Hora', color='purple', alpha=0.5)
    plt.plot(media_movel_comentarios.index, media_movel_comentarios, label=f'Média Móvel ({janela_media_movel}h)', color='red')
    plt.axhline(y=limiar_alta_comentarios, color='green', linestyle='--', label='Limiar Alta Atividade')
    plt.axhline(y=limiar_baixa_comentarios, color='orange', linestyle='--', label='Limiar Baixa Atividade')
    plt.title('Série Temporal de Comentários por Hora')
    plt.xlabel('Data e Hora')
    plt.ylabel('Número de Comentários')
    plt.legend()
    plt.savefig('serie_temporal_comentarios.png')
    plt.close()

# Gráfico 3: Sazonalidade (Média por Hora do Dia)
plt.figure(figsize=(15, 5))
plt.plot(serie_posts_sazonal.index, serie_posts_sazonal, label='Publicações (Média por Hora do Dia)', color='blue')
if not serie_comentarios_sazonal.empty:
    plt.plot(serie_comentarios_sazonal.index, serie_comentarios_sazonal, label='Comentários (Média por Hora do Dia)', color='purple')
plt.title('Sazonalidade - Média por Hora do Dia')
plt.xlabel('Hora do Dia')
plt.ylabel('Média de Atividade')
plt.legend()
plt.savefig('sazonalidade_hora_dia.png')
plt.close()

# Identifica picos de atividade
picos_posts = serie_posts[serie_posts >= limiar_alta_posts].sort_values(ascending=False)
picos_comentarios = serie_comentarios[serie_comentarios >= limiar_alta_comentarios].sort_values(ascending=False) if not serie_comentarios.empty else pd.Series()

print("\nPrincipais Picos de Publicações (Top 5):")
print(picos_posts.head(5))
if not picos_comentarios.empty:
    print("\nPrincipais Picos de Comentários (Top 5):")
    print(picos_comentarios.head(5))

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
\fancyhead[C]{Análise Temporal - Instagram Outubro 2018}
\fancyfoot[C]{\thepage}

\titleformat{\section}{\normalfont\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\normalfont\large\bfseries}{\thesubsection}{1em}{}

\begin{document}

\begin{titlepage}
    \centering
    \vspace*{2cm}
    {\Huge\bfseries Análise Temporal de Publicações e Comentários no Instagram\\Outubro 2018\par}
    \vspace{1cm}
    {\Large Israel - DoE Atividade 1\par}
    \vspace{2cm}
    {\large \today\par}
    \vspace{2cm}
\end{titlepage}

\section{Série Temporal de Publicações por Hora}
\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{serie_temporal_publicacoes.png}
    \caption{Série temporal de publicações por hora, com média móvel de 24 horas e limiares de alta e baixa atividade.}
\end{figure}
\clearpage

\section{Série Temporal de Comentários por Hora}
"""
if not serie_comentarios.empty:
    latex_content += r"""
    \begin{figure}[h]
        \centering
        \includegraphics[width=\textwidth]{serie_temporal_comentarios.png}
        \caption{Série temporal de comentários por hora, com média móvel de 24 horas e limiares de alta e baixa atividade.}
    \end{figure}
    """
else:
    latex_content += r"""
    \subsection*{Nota}
    Não há dados suficientes para a série temporal de comentários.
    """

latex_content += r"""
\clearpage

\section{Sazonalidade - Média por Hora do Dia}
\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{sazonalidade_hora_dia.png}
    \caption{Média de publicações e comentários por hora do dia, mostrando padrões sazonais.}
\end{figure}
\clearpage

\end{document}
"""

# Salva o documento LaTeX sem compilar
with open("analise_temporal.tex", "w") as f:
    f.write(latex_content)

print("Documento LaTeX gerado: analise_temporal.tex")
print("Compile manualmente com pdflatex ou online (e.g., Overleaf) para gerar o PDF.")