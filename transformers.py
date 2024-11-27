import os
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import PyPDF2

# 1. Função para extrair texto dos PDFs e dividi-lo em capítulos/páginas
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        texts = [page.extract_text() for page in reader.pages]
        return texts  # Lista de textos por página

# 2. Classe para gerar embeddings com Transformers
class TextEmbedder:
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"):   #'1 modelo a ser testado- multi-qa-mpnet-base-dot-v1 , all-mpnet-base-v2 , multi-qa-distilbert-cos-v1
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embeddings(self, texts):
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
def process_pdfs(pdf_folder, embedder):
    all_texts = []
    pdf_sources = []  # Rastreia de qual PDF cada trecho vem
    embeddings = []

    # Percorrer todos os PDFs na pasta
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            texts = extract_text_from_pdf(pdf_path)
            all_texts.extend(texts)
            pdf_sources.extend([pdf_file] * len(texts))

    # Gerar embeddings para todos os textos
    print("Gerando embeddings...")
    embeddings = embedder.get_embeddings(all_texts)
    print("Embeddings gerados com sucesso!")
    return all_texts, pdf_sources, embeddings

# 4. Função para buscar a partir de uma consulta
def search_documents(query, all_texts, pdf_sources, embeddings, embedder):
    # Gerar embedding da consulta
    print("Gerando embedding da consulta...")
    query_embedding = embedder.get_embeddings([query])

    # Calcular similaridade
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    results = sorted(
        enumerate(similarities),
        key=lambda x: x[1],
        reverse=True
    )

    # Retornar o resultado mais relevante
    if results:
        idx, score = results[0]  # Apenas o mais relevante
        return {
            "pdf": pdf_sources[idx],
            "page": idx + 1,  # Página simulada (depende de divisão)
            "similarity": score,
            "text": all_texts[idx]  # Texto completo
        }
    return None

# 5. Função para salvar a resposta no formato desejado
def save_response_to_file(response, output_file, user_query):
    if response:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Pergunta: {user_query}\n\n")
            f.write("Resposta:\n")
            f.write(response['text'])
            f.write("\n\nReferência:\n")
            f.write(f"PDF: {response['pdf']}, Página: {response['page']}\n")
        print(f"Resposta salva em {output_file}")
    else:
        print("Nenhuma resposta relevante encontrada.")

# 6. Main
if __name__ == "__main__":
    pdf_folder = "C:\\Documents"  # Substitua pelo caminho da sua pasta com PDFs
    output_file = "resposta.txt"  # Arquivo para salvar a resposta
    embedder = TextEmbedder()

    # Processar PDFs e criar embeddings
    all_texts, pdf_sources, embeddings = process_pdfs(pdf_folder, embedder)

    # Solicitar consulta ao usuário
    user_query = input("Digite sua pergunta ou descrição do que você procura: ")

    # Buscar documento mais relevante
    response = search_documents(user_query, all_texts, pdf_sources, embeddings, embedder)

    # Salvar a resposta no arquivo
    save_response_to_file(response, output_file, user_query)
