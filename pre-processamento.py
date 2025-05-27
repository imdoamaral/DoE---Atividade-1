'''
Esta análise focará num subconjunto específico dos dados, como sugerido no próprio enunciado.

O foco será no primeiro turno das eleições (outubro de 2018) para reduzir a carga de memória.
'''

import pandas as pd
import os
from datetime import datetime
import glob
import csv

# Define o diretorio onde os arquivos .csv estao localizados
diretorio_dados = "/home/israel/Downloads/Instagram Data/smartdata_ig/data/BR/"

# Define a lista de arquivos .csv para o periodo de outubro de 2018
arquivos_outubro_2018 = glob.glob(os.path.join(diretorio_dados, "2018-10*.csv"))

# Funçao para converter timestamps para datetime
def converter_timestamp_para_datetime(timestamp):
    try:
        return pd.to_datetime(timestamp, unit='s')
    except:
        return pd.NaT # retorna NaT para valores invalidos
    
def pre_processar_chunk(chunk):

    # Converte colunas de data
    chunk['created_time'] = chunk['created_time'].apply(converter_timestamp_para_datetime)
    chunk['created_time_comment'] = chunk['created_time_comment'].apply(converter_timestamp_para_datetime)

    # Remove duplicatas
    chunk = chunk.drop_duplicates()

    # Trata valores ausentes (substituir por NaN ou remover)
    chunk = chunk.replace('', pd.NA)
    chunk = chunk.dropna(subset=['created_time', 'media_owner_id', 'comment_id'], how='any') # Remove linhas com valores ausentes em colunas criticas

    return chunk

# Lista para armazenar os chunks processados
dados_processados = []

# Tamanho do chunk para gerenciar memoria (monitorar com o htop pra ver se dar pra liberar mais)
tamanho_chunk = 5000

# itera sobre os arquikvos de outubro de 2018
for arquivo in arquivos_outubro_2018:
    print(f"Processando arquivo: {arquivo}")
    try:
        # Le o arquivos em chunks com aplicando os parametros abaixo
        for chunk in pd.read_csv(
            arquivo,
            chunksize=tamanho_chunk,
            encoding='utf-8',
            quoting=csv.QUOTE_ALL, # Força aspas "" em todos os campos
            on_bad_lines='skip', # Pula linhas malformadas
            low_memory=False, # Garante que erros nao interrompam o processamento
            escapechar='\\', # Para lidar com virgulas ou aspas em campos de texto
            doublequote=True # Trata aspas duplas corretamente
        ):
            chunk_processado = pre_processar_chunk(chunk)
            dados_processados.append(chunk_processado)
    except Exception as error:
        print(f"Erro ao processar {arquivo}: {str(error)}")
        continue

# Concatena chunks
if dados_processados:
    dados_completos = pd.concat(dados_processados, ignore_index=True)
    caminho_raiz = os.path.dirname(os.path.abspath(__file__))
    caminho_saida = os.path.join(caminho_raiz, "dados_pre_processados_outubro_2018.csv")
    dados_completos.to_csv(caminho_saida, index=False)
    print(f"Dados pre-processados salvos em: {caminho_saida}")
    print("\nInformaçoes do conjunto de dados processados:")
    print(dados_completos.info())
else:
    print("Nenhum dado processado. Verifique os arquivos de entrada.")