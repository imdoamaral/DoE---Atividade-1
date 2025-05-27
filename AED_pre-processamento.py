import pandas as pd
import os
import glob
import csv

# Define o diretório onde os arquivos .csv estão localizados
diretorio_dados = "/home/israel/Downloads/Instagram Data/smartdata_ig/data/BR/"

# Define a lista de arquivos .csv para 7 de outubro de 2018
arquivos_outubro_2018 = glob.glob(os.path.join(diretorio_dados, "2018-10-07.csv"))

# Função para converter timestamps ou datetimes
def converter_data(valor):
    try:
        # Se for timestamp numérico (UNIX)
        if isinstance(valor, (int, float)) or (isinstance(valor, str) and valor.isdigit()):
            return pd.to_datetime(int(valor), unit='s')
        # Se for string no formato datetime
        return pd.to_datetime(valor, errors='coerce')
    except:
        return pd.NaT

def pre_processar_chunk(chunk):
    # Converte colunas de data
    chunk['created_time'] = chunk['created_time'].apply(converter_data)
    chunk['created_time_comment'] = chunk['created_time_comment'].apply(converter_data)

    # Remove duplicatas completas (linhas inteiras iguais)
    chunk = chunk.drop_duplicates()

    # Substitui strings vazias por NaN
    chunk = chunk.replace('', pd.NA)

    # Remove linhas sem created_time ou media_owner_id
    chunk = chunk.dropna(subset=['created_time', 'media_owner_id'], how='any')

    return chunk

# Lista para armazenar os chunks processados
dados_processados = []

# Tamanho do chunk
tamanho_chunk = 100000

# Itera sobre os arquivos de 7 de outubro de 2018
chunk_count = 0
for arquivo in arquivos_outubro_2018:
    print(f"Processando arquivo: {arquivo}")
    try:
        for chunk in pd.read_csv(
            arquivo,
            chunksize=tamanho_chunk,
            encoding='utf-8',
            quoting=csv.QUOTE_ALL,
            on_bad_lines='skip',
            low_memory=False,
            escapechar='\\',
            doublequote=True
        ):
            chunk_processado = pre_processar_chunk(chunk)
            print(f"Chunk {chunk_count + 1}: {len(chunk_processado)} linhas processadas")
            dados_processados.append(chunk_processado)
            chunk_count += 1
    except Exception as error:
        print(f"Erro ao processar {arquivo}: {str(error)}")
        continue

# Concatena chunks
if dados_processados:
    dados_completos = pd.concat(dados_processados, ignore_index=True)
    caminho_raiz = os.path.dirname(os.path.abspath(__file__))
    caminho_saida = os.path.join(caminho_raiz, "dados_pre_processados_outubro_2018.csv")
    dados_completos.to_csv(caminho_saida, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
    print(f"Dados pré-processados salvos em: {caminho_saida}")
    print("\nInformações do conjunto de dados processados:")
    print(dados_completos.info())
    print(f"\nValores nulos por coluna:\n{dados_completos.isna().sum()}")
else:
    print("Nenhum dado processado. Verifique os arquivos de entrada.")