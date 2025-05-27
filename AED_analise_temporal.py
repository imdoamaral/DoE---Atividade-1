import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

# Define o caminho do arquivo pré-processado
caminho_arquivo = "/home/israel/vscode/DoE - Atividade 1/dados_pre_processados_outubro_2018.csv"

# Verifica se o arquivo existe
if not os.path.exists(caminho_arquivo):
    print(f"Erro: Arquivo '{caminho_arquivo}' não encontrado. Reexecute o script de pré-processamento.")
    exit()

# Define o tamanho do chunk
tamanho_chunk = 10000

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
    
    # Extrai hora completa (data + hora)
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
        posts, comentarios = processar_chunk_temporal(chunk, chunk_count)
        serie_posts = serie_posts.add(posts, fill_value=0)
        serie_comentarios = serie_comentarios.add(comentarios, fill_value=0)
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

# Reindexa para garantir continuidade, restrito a 7 de outubro de 2018
data_inicio = pd.Timestamp('2018-10-07 00:00:00')
data_fim = pd.Timestamp('2018-10-07 23:00:00')
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

# Adiciona log para inspeção
print("\nSazonalidade Publicações (Média por Hora do Dia):")
print(serie_posts_sazonal)
print("\nSazonalidade Comentários (Média por Hora do Dia):")
print(serie_comentarios_sazonal)

# Gera os gráficos e salva como PNG temporariamente
# Gráfico 1: Séries Temporais de Publicações
plt.figure(figsize=(12, 9))  # Proporção 4:3
plt.plot(serie_posts.index.strftime('%H:%M'), serie_posts, label='Publicações por Hora', color='blue', alpha=0.5)
plt.plot(media_movel_posts.index.strftime('%H:%M'), media_movel_posts, label=f'Média Móvel ({janela_media_movel}h)', color='red')
plt.axhline(y=limiar_alta_posts, color='green', linestyle='--', label='Limiar Alta Atividade (Pub.)')
plt.axhline(y=limiar_baixa_posts, color='orange', linestyle='--', label='Limiar Baixa Atividade (Pub.)')
plt.title('Série Temporal de Publicações por Hora (7 de Outubro de 2018)')
plt.xlabel('Hora do Dia')
plt.ylabel('Número de Publicações')
plt.legend()
plt.tight_layout()
plt.savefig('serie_temporal_publicacoes.png')
plt.close()

# Gráfico 2: Séries Temporais de Comentários
if not serie_comentarios.empty:
    plt.figure(figsize=(12, 9))  # Proporção 4:3
    plt.plot(serie_comentarios.index.strftime('%H:%M'), serie_comentarios, label='Comentários por Hora', color='purple', alpha=0.5)
    plt.plot(media_movel_comentarios.index.strftime('%H:%M'), media_movel_comentarios, label=f'Média Móvel ({janela_media_movel}h)', color='red')
    plt.axhline(y=limiar_alta_comentarios, color='green', linestyle='--', label='Limiar Alta Atividade (Com.)')
    plt.axhline(y=limiar_baixa_comentarios, color='orange', linestyle='--', label='Limiar Baixa Atividade (Com.)')
    plt.title('Série Temporal de Comentários por Hora (7 de Outubro de 2018)')
    plt.xlabel('Hora do Dia')
    plt.ylabel('Número de Comentários')
    plt.legend()
    plt.tight_layout()
    plt.savefig('serie_temporal_comentarios.png')
    plt.close()

# Gráfico 3: Sazonalidade (Média por Hora do Dia)
fig, ax1 = plt.subplots(figsize=(12, 9))  # Proporção 4:3
horas = range(24)  # Índices de 0 a 23

# Eixo principal para publicações
ax1.plot(horas, serie_posts_sazonal.values, label='Publicações (Média por Hora)', color='blue')
ax1.axhline(y=limiar_alta_posts, color='green', linestyle='--', label='Limiar Alta (Pub.)')
ax1.axhline(y=limiar_baixa_posts, color='orange', linestyle='--', label='Limiar Baixa (Pub.)')
ax1.set_title('Sazonalidade - Média por Hora do Dia (7 de Outubro 2018)')
ax1.set_xlabel('Hora do Dia')
ax1.set_ylabel('Média de Publicações', color='blue')
ax1.set_xticks(range(0, 24, 1))  # Mostra todas as horas de 0 a 23
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(0, 250)  # Escala para publicações (máximo 240)

# Eixo secundário para comentários
ax2 = ax1.twinx()
ax2.plot(horas, serie_comentarios_sazonal.values, label='Comentários (Média por Hora)', color='purple')
ax2.axhline(y=limiar_alta_comentarios, color='green', linestyle='-.', label='Limiar Alta (Com.)')
ax2.axhline(y=limiar_baixa_comentarios, color='orange', linestyle='-.', label='Limiar Baixa (Com.)')
ax2.set_ylabel('Média de Comentários', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')
ax2.set_ylim(0, 50000)  # Escala para comentários (máximo 48.415)

# Ajuste da legenda combinada
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.tight_layout()
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
\fancyhead[C]{Análise Temporal - Instagram 7 de Outubro 2018}
\fancyfoot[C]{\thepage}

\titleformat{\section}{\normalfont\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\normalfont\large\bfseries}{\thesubsection}{1em}{}

\begin{document}

\begin{titlepage}
    \centering
    \vspace*{2cm}
    {\Huge\bfseries Análise Temporal de Publicações e Comentários no Instagram\\7 de Outubro 2018\par}
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