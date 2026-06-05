import csv
import os
from pathlib import Path
from dotenv import load_dotenv
from src.models import Document
from src.chunking import RecursiveChunker
from src.store import EmbeddingStore
from src.embeddings import OpenAIEmbedder

# Nạp các biến môi trường từ file .env
load_dotenv(override=False)

# 1. Định nghĩa phím từ khóa kiểm định tương ứng với 10 câu hỏi để tự động xác minh kết quả
KEYWORDS_MAP = {
    1: ["VGC", "HNX"],
    2: ["Khánh Hòa", "29"],
    3: ["Goldman", "2.300"],
    4: ["hành trình", "an toàn"],
    5: ["đèo Cả"],
    6: ["83,5"],
    7: ["MBV"],
    8: ["biệt thự", "liền kề"],
    9: ["Quảng Ngãi", "Đăk Re"],
    10: ["Cơ sở dữ liệu", "dân cư"]
}

def parse_benchmark_qa(file_path: str = "data/benchmark_qa.txt") -> list[dict]:
    """Phân tích file benchmark_qa.txt thành danh sách câu hỏi và câu trả lời chuẩn."""
    import re
    if not Path(file_path).exists():
        print(f"Lỗi: Không tìm thấy file {file_path}!")
        return []
        
    content = Path(file_path).read_text(encoding="utf-8")
    questions = re.findall(r"Câu hỏi (\d+): (.*?)\nTài liệu đích: (.*?)\nCâu trả lời đúng: (.*?)\n", content)
    
    qa_list = []
    for q_id_str, query, source, gold in questions:
        q_id = int(q_id_str)
        qa_list.append({
            "id": q_id,
            "query": query.strip(),
            "gold_answer": gold.strip(),
            "keywords": KEYWORDS_MAP.get(q_id, [])
        })
    return qa_list

BENCHMARK_QUERIES = parse_benchmark_qa()

def load_raw_csv_data(csv_path: str = "data/100_data.csv") -> list[dict]:
    """Đọc dữ liệu thô từ file CSV."""
    data = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def run_experiment(chunk_size: int, embedder):
    print(f"\n=======================================================")
    print(f" BẮT ĐẦU THỬ NGHIỆM: RecursiveChunker (chunk_size={chunk_size})")
    print(f"=======================================================")
    
    # Khởi tạo chunker đệ quy
    chunker = RecursiveChunker(chunk_size=chunk_size)
    
    raw_rows = load_raw_csv_data()
    chunked_documents = []
    
    # Thực hiện chia nhỏ dữ liệu đệ quy
    chunk_counter = 0
    for idx, row in enumerate(raw_rows):
        content = row.get("content", "")
        chunks = chunker.chunk(content)
        
        for sub_idx, chunk_text in enumerate(chunks):
            metadata = {
                "source_url": row.get("url", ""),
                "title": row.get("title", ""),
                "category": row.get("category", ""),
                "time": row.get("time", ""),
                "chunk_index": sub_idx,
                "doc_id": f"doc_{idx}"
            }
            doc = Document(
                id=f"chunk_{chunk_counter}",
                content=chunk_text,
                metadata=metadata
            )
            chunked_documents.append(doc)
            chunk_counter += 1
            
    print(f"-> Tổng số bài báo gốc: {len(raw_rows)}")
    print(f"-> Tổng số chunks được sinh ra: {len(chunked_documents)}")
    
    # Lưu vào EmbeddingStore sử dụng OpenAI API
    store = EmbeddingStore(collection_name=f"test_recursive_{chunk_size}", embedding_fn=embedder)
    store.add_documents(chunked_documents)
    
    # Thực hiện chạy thử nghiệm 5 câu hỏi benchmark
    correct_retrievals = 0
    for q_item in BENCHMARK_QUERIES:
        q_id = q_item["id"]
        query = q_item["query"]
        gold = q_item["gold_answer"]
        
        print(f"\n[Q{q_id}] Câu hỏi: {query}")
        print(f"    Câu trả lời chuẩn: {gold}")
        
        results = store.search(query, top_k=3)
        print(f"    Kết quả truy xuất (Top 3):")
        
        found_in_top_3 = False
        for rank, res in enumerate(results, start=1):
            score = res["score"]
            title = res["metadata"].get("title", "Unknown")
            category = res["metadata"].get("category", "Unknown")
            preview = res["content"][:100].replace('\n', ' ')
            
            # Kiểm tra xem từ khóa đặc trưng của câu trả lời chuẩn có nằm trong chunk không
            keywords = q_item.get("keywords", [])
            
            match = all(k.lower() in res["content"].lower() for k in keywords)
            status_indicator = "✅ [KHỚP NGỮ CẢNH CHUẨN]" if match else "❌"
            if match:
                found_in_top_3 = True
                
            print(f"      {rank}. Score: {score:.4f} | Cat: {category} | Tiêu đề: {title} {status_indicator}")
            print(f"         Nội dung preview: \"{preview}...\"")
            
        if found_in_top_3:
            correct_retrievals += 1
            
    accuracy = (correct_retrievals / len(BENCHMARK_QUERIES)) * 100
    print(f"\n=== KẾT QUẢ TỔNG HỢP CHO CHUNK_SIZE={chunk_size} ===")
    print(f"- Số câu truy xuất thành công (Top-3 chứa ngữ cảnh chứa Gold Answer): {correct_retrievals}/{len(BENCHMARK_QUERIES)}")
    print(f"- Độ chính xác Retrieval (Accuracy): {accuracy}%")
    return accuracy, len(chunked_documents)

if __name__ == "__main__":
    # Khởi tạo OpenAI Embedder
    try:
        embedder = OpenAIEmbedder()
        print(f"Khởi tạo thành công OpenAI Embedder ({embedder.model_name})")
    except Exception as e:
        print(f"Lỗi khởi tạo OpenAI Embedder: {e}")
        exit(1)

    configs = [600, 1200]
    summary = {}
    for size in configs:
        acc, num_chunks = run_experiment(size, embedder)
        summary[size] = {"accuracy": acc, "num_chunks": num_chunks}
        
    print("\n=======================================================")
    print(" BẢNG SO SÁNH KẾT QUẢ CHUNK SIZE (RECURSIVE SPLITTING)")
    print("=======================================================")
    print(f"{'Chunk Size':<12} | {'Tổng số Chunks':<16} | {'Độ chính xác RAG':<16}")
    print("-" * 52)
    for size, stats in summary.items():
        print(f"{size:<12} | {stats['num_chunks']:<16} | {stats['accuracy']:.1f}%")
    print("=======================================================")
