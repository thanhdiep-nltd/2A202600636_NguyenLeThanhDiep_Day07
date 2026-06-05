import csv
import sys
from pathlib import Path
from dotenv import load_dotenv

from src.models import Document
from src.chunking import RecursiveChunker
from src.store import EmbeddingStore
from src.embeddings import OpenAIEmbedder
from src.agent import KnowledgeBaseAgent

# Nạp các biến môi trường
load_dotenv(override=False)

def load_documents_from_csv(csv_path: str = "data/100_data.csv") -> list[Document]:
    """Đọc dữ liệu thô từ file CSV."""
    documents = []
    if not Path(csv_path).exists():
        print(f"Lỗi: Không tìm thấy file {csv_path}!")
        sys.exit(1)
        
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            metadata = {
                "source_url": row.get("url", ""),
                "title": row.get("title", ""),
                "category": row.get("category", ""),
                "time": row.get("time", ""),
                "doc_id": f"doc_{i}"
            }
            doc = Document(
                id=f"doc_{i}",
                content=row.get("content", ""),
                metadata=metadata
            )
            documents.append(doc)
    return documents

def main():
    print("=== KHỞI ĐỘNG HỆ THỐNG RAG INTERACTIVE CHAT ===")
    
    # 1. Khởi tạo Embedder
    try:
        embedder = OpenAIEmbedder()
        print(f"[*] Khởi tạo thành công OpenAI Embedder ({embedder.model_name})")
    except Exception as e:
        print(f"Lỗi cấu hình OpenAI API: {e}")
        print("Vui lòng kiểm tra biến OPENAI_API_KEY trong file .env")
        sys.exit(1)
        
    # 2. Chọn cấu hình chunk size
    print("\nChọn kích thước chunk (chunk_size) để thử nghiệm:")
    print("1. 600 ký tự (mặc định)")
    print("2. 1200 ký tự")
    choice = input("Lựa chọn của bạn (1 hoặc 2): ").strip()
    chunk_size = 1200 if choice == "2" else 600
    
    # 3. Nạp dữ liệu & Thực hiện chunking
    print(f"\n[*] Đang đọc file CSV và phân tách đệ quy (chunk_size={chunk_size})...")
    raw_docs = load_documents_from_csv()
    chunker = RecursiveChunker(chunk_size=chunk_size)
    
    chunked_documents = []
    chunk_counter = 0
    for idx, doc in enumerate(raw_docs):
        chunks = chunker.chunk(doc.content)
        for sub_idx, chunk_text in enumerate(chunks):
            # Kế thừa metadata từ tài liệu gốc
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = sub_idx
            
            chunk_doc = Document(
                id=f"chunk_{chunk_counter}",
                content=chunk_text,
                metadata=metadata
            )
            chunked_documents.append(chunk_doc)
            chunk_counter += 1
            
    print(f"-> Tổng số bài báo gốc: {len(raw_docs)}")
    print(f"-> Phân tách thành {len(chunked_documents)} chunks.")
    
    # 4. Đưa dữ liệu vào Store và sinh Vector Embeddings
    store = EmbeddingStore(collection_name=f"interactive_store_{chunk_size}", embedding_fn=embedder)
    if store.get_collection_size() == 0:
        print("\n[*] Cơ sở dữ liệu trống. Đang tạo embeddings qua OpenAI API (vui lòng chờ trong giây lát)...")
        store.add_documents(chunked_documents)
        print("[+] Đã nạp thành công dữ liệu vào vector store!")
    else:
        print(f"\n[+] Phát hiện dữ liệu cũ trong ChromaDB (Kích thước: {store.get_collection_size()} chunks). Bỏ qua bước tạo embeddings mới!")
    
    # 5. Khởi tạo Agent
    # Ở đây chúng ta mock LLM đơn giản để in ra các thông tin cần thiết
    def simple_rag_llm(prompt: str) -> str:
        return "[LLM Trả lời] (Thông tin đã được truy xuất thành công từ ngữ cảnh ở bên dưới)."

    agent = KnowledgeBaseAgent(store=store, llm_fn=simple_rag_llm)
    
    print("\n=========================================================================")
    print("HỆ THỐNG ĐÃ SẴN SÀNG. Gõ 'exit' hoặc 'quit' để thoát chương trình.")
    print("=========================================================================")
    
    while True:
        try:
            query = input("\nNhập câu hỏi của bạn (Chat): ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("Tạm biệt!")
                break
                
            print("\n" + "-"*50)
            print(f"[*] Đang thực hiện RAG cho câu hỏi: '{query}'...")
            
            # Thực hiện tìm kiếm chunks liên quan
            results = store.search(query, top_k=3)
            
            print("\n[Kết quả truy xuất - Top 3 Chunks liên quan nhất]:")
            for rank, res in enumerate(results, start=1):
                score = res["score"]
                title = res["metadata"].get("title", "Không rõ")
                category = res["metadata"].get("category", "Không rõ")
                content = res["content"]
                
                print(f"\n{rank}. [Độ tương đồng: {score:.4f}] | Chủ đề: {category}")
                print(f"   Tiêu đề bài báo: {title}")
                print(f"   Nội dung chunk: \"{content}\"")
                print("-" * 30)
                
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break

if __name__ == "__main__":
    main()
