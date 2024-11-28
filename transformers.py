import os
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import torch
import PyPDF2


# 1. Função para extrair texto dos PDFs e dividi-lo em capítulos/páginas
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        texts = [reader.pages[i].extract_text() for i in range(len(reader.pages))]
        return texts  # Lista de textos por página



# 2. Classe para gerar embeddings com Transformers
class TextEmbedder:
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"):
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
    print("Gerando embeddings...")
    embeddings = embedder.get_embeddings(all_texts)
    print("Embeddings gerados com sucesso!")
    return all_texts, pdf_sources, page_numbers, embeddings

# 4. Função para buscar a partir de uma consulta
def search_documents(query, all_texts, pdf_sources, page_numbers, embeddings, embedder, top_k=3):
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
    
    return top_results

# 5. Função para salvar a resposta no formato desejado
def save_response_to_file(response, output_file, user_query):
    if response:
        # Acessa o primeiro item da lista de resultados
        result = response[0] if isinstance(response, list) else response

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Pergunta: {user_query}\n\n")
            f.write("Resposta:\n")
            f.write(result['text'])  # Acessa o texto da resposta
            f.write("\n\nReferência:\n")
            f.write(f"PDF: {result['pdf']}, Página: {result['page']}\n")  # Agora tem a chave 'page'
        print(f"Resposta salva em {output_file}")
    else:
        print("Nenhuma resposta relevante encontrada.")

# 6. Main
if __name__ == "__main__":
    pdf_folder = "C:\\Users\\Win10\\Documents\\PDFS"  # Substitua pelo caminho da sua pasta com PDFs
    output_file = "resposta.txt"  # Arquivo para salvar a resposta
    embedder = TextEmbedder()

    # Processar PDFs e criar embeddings
    all_texts, pdf_sources, page_numbers, embeddings = process_pdfs(pdf_folder, embedder)

    # Solicitar consulta ao usuário
    user_query = input("Digite sua pergunta ou descrição do que você procura: ")

    # Buscar documento mais relevante
    response = search_documents(user_query, all_texts, pdf_sources, page_numbers, embeddings, embedder)

    # Salvar a resposta no arquivo
    save_response_to_file(response, output_file, user_query)
