import os
import psutil
from datetime import datetime
import json
import torch
import PyPDF2
from typing import List
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, WebSocket

app = FastAPI()

# Função para logar o uso exclusivo de recursos do processo
def log_process_usage(log_file="process_usage.log"):
    # Pegar o ID do processo atual (o processo que está rodando o código)
    process = psutil.Process(os.getpid())
    
    # Obter o uso de CPU e memória do processo específico
    cpu_percent = process.cpu_percent(interval=1)  # Uso de CPU do processo
    memory_info = process.memory_info()  # Informações sobre o uso de memória
    memory_percent = memory_info.rss / (1024 * 1024)  # Em MB (RSS = resident set size)

    # Logar essas informações em um arquivo
    with open(log_file, "a", encoding="utf-8") as log:
        data = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"{data} - CPU do Processo: {cpu_percent}% - Memória do Processo: {memory_percent:.2f} MB\n")

# 1. Função para extrair texto dos PDFs e dividi-lo em capítulos/páginas
def extract_text_from_pdf(pdf_path: str) -> List[str]:
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        texts = [reader.pages[i].extract_text() for i in range(len(reader.pages))]
        return texts  # Lista de textos por página

# 2. Classe para gerar embeddings com Transformers
class TextEmbedder:
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embeddings(self, texts: List[str]):
        # Tokenizar os textos
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        # Obter os embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

# 3. Função para processar PDFs em uma pasta e criar embeddings
def process_pdfs(pdf_folder: str, embedder: TextEmbedder):
    all_texts = []
    pdf_sources = []  # Rastreia de qual PDF cada trecho vem
    page_numbers = []  # Rastreia o número da página para cada trecho
    embeddings = []

    # Percorrer todos os PDFs na pasta
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            texts = extract_text_from_pdf(pdf_path)
            # Adicionar o texto e o número da página
            for i, text in enumerate(texts):
                all_texts.append(text)
                pdf_sources.append(pdf_file)
                page_numbers.append(i + 1)  # Páginas começam do número 1

    # Gerar embeddings para todos os textos
    print("Gerando embeddings...")  # Aviso no terminal
    embeddings = embedder.get_embeddings(all_texts)
    print("Embeddings gerados com sucesso!")
    return all_texts, pdf_sources, page_numbers, embeddings

# 4. Função para buscar a partir de uma consulta
def search_documents(query: str, all_texts: List[str], pdf_sources: List[str], page_numbers: List[int], embeddings, embedder: TextEmbedder, top_k=3):
    # Gerar embedding da consulta
    query_embedding = embedder.get_embeddings([query])

    # Calcular similaridade
    similarities = cosine_similarity(query_embedding, embeddings).flatten()

    # Ordenar e pegar os melhores top_k resultados
    results = sorted(
        enumerate(similarities),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    top_results = []
    for idx, score in results:
        top_results.append({
            "pdf": pdf_sources[idx],
            "text": all_texts[idx],
            "page": page_numbers[idx],  # Adicionando o número da página
            "similarity": score
        })

    return convert_to_serializable(top_results)

# Função para garantir que o tipo dos dados seja serializável
def convert_to_serializable(response):
    # Converte todos os embeddings para float padrão
    for item in response:
        item['similarity'] = float(item['similarity'])  # Garante que o valor seja float padrão
        item['text'] = str(item['text'])  # Garantir que o texto seja string
    return response


# Variáveis globais para manter os dados em memória
all_texts = []
pdf_sources = []
page_numbers = []
embeddings = []

embedder = TextEmbedder()

@app.on_event("startup")
async def startup_event():
    # Processar PDFs e criar embeddings uma vez, quando o servidor iniciar
    pdf_folder = "C:\\Users\\Win10\\Documents\\ImagoDevProjetos\\TransformersSearch\\api\\files\\PDFS"  # Substitua pelo caminho da sua pasta com PDFs
    global all_texts, pdf_sources, page_numbers, embeddings
    all_texts, pdf_sources, page_numbers, embeddings = process_pdfs(pdf_folder, embedder)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        # Receber a consulta do cliente
        user_query = await websocket.receive_text()
        print(f"Consulta recebida: {user_query}")

        # Buscar documento mais relevante
        response = search_documents(user_query, all_texts, pdf_sources, page_numbers, embeddings, embedder)
        
        if response:
            # Logar o uso de recursos do processo
            log_process_usage()

            output_file = 'teste.txt'
            data = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(output_file, "w", encoding="utf-8") as f:
                # Escrever a data e a pergunta
                f.write(f"Dia da Consulta: {data}\n")
                f.write(f"Pergunta: {user_query}\n")
                
                # Converter a resposta para string (JSON)
                response_str = json.dumps(response, indent=4, ensure_ascii=False)
                
                # Escrever a resposta formatada no arquivo
                f.write("Resposta:\n")
                f.write(response_str)
                
            # Enviar a resposta com os documentos encontrados
            await websocket.send_json(response)
        else:
            await websocket.send_text("Nenhuma resposta relevante encontrada.")
