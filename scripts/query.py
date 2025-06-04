import os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
# from langchain.document_loaders import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

def main():
    # 1. Load local embeddings model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # 2. Load FAISS index (with the same Sentence-Transformer embeddings wrapper)
    from langchain.embeddings.base import Embeddings
    class SBERTEmbeddings(Embeddings):
        def __init__(self, model):
            self.model = model

        def embed_documents(self, texts):
            return [self.model.encode(t) for t in texts]

        def embed_query(self, text):
            return self.model.encode(text)

    sbert = SBERTEmbeddings(embedder)

    db = FAISS.load_local("faiss_index", sbert, allow_dangerous_deserialization=True)

    # 3. Load a local seq2seq LLM (flan-t5-small) for generation
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    # 4. CLI loop for RAG-style Q&A
    print("📚 PDF Q&A CLI (using local open-source LLM; type 'exit' to quit)")
    while True:
        query = input("\n🔎 Ask a question: ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        # 4a. Retrieve top-k chunks
        docs_and_scores = db.as_retriever(search_kwargs={"k": 2}).get_relevant_documents(query)
        retrieved_texts = [doc.page_content for doc in docs_and_scores]

        # 4b. Build a single prompt combining retrieved chunks + user query
        prompt = "Answer the question based on the following passages:\n\n"
        for idx, chunk in enumerate(retrieved_texts, start=1):
            prompt += f"Passage {idx}:\n{chunk}\n\n"
        prompt += f"Question: {query}"

        # 4c. Generate the answer
        output = gen_pipeline(prompt, max_length=256, do_sample=False)
        answer = output[0]["generated_text"]

        print("\n📝 Answer:\n", answer)

if __name__ == "__main__":
    main()
