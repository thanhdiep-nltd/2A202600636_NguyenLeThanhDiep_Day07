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

# 1. Định nghĩa 5 câu hỏi chuẩn và câu trả lời chuẩn tương ứng
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Vào phiên giao dịch ngày 03/03/2023, chỉ số VN-Index đóng cửa giảm bao nhiêu phần trăm và rổ VN30 có duy nhất mã cổ phiếu nào tăng giá?",
        "gold_answer": "VN-Index đóng cửa giảm 1,24% và mã cổ phiếu tăng giá duy nhất trong rổ VN30 là PLX (tăng 0,39%)."
    },
    {
        "id": 2,
        "query": "Panasonic đã đầu tư bao nhiêu USD vào nhà máy mới tại Bình Dương và năng suất dự kiến vào năm 2025 là bao nhiêu?",
        "gold_answer": "Vốn đầu tư khoảng 45 triệu USD và năng suất dự kiến đạt khoảng 3 triệu sản phẩm vào năm 2025."
    },
    {
        "id": 3,
        "query": "Bộ Tài chính đề xuất ưu đãi thuế thu nhập doanh nghiệp bao nhiêu phần trăm đối với cơ quan báo chí ngoài báo in?",
        "gold_answer": "Bộ Tài chính đề xuất thuế suất ưu đãi 15% đối với thu nhập của các cơ quan báo chí ngoài báo in."
    },
    {
        "id": 4,
        "query": "Theo dự báo của IATA, thị trường hàng không thế giới sẽ phục hồi vượt mức trước dịch vào năm nào?",
        "gold_answer": "IATA dự báo thị trường hàng không thế giới sẽ phục hồi vượt mức trước dịch Covid-19 vào đầu năm 2024."
    },
    {
        "id": 5,
        "query": "Khu du lịch quốc gia Ninh Chữ dự kiến đón bao nhiêu lượt khách du lịch và bao nhiêu khách quốc tế vào năm 2030?",
        "gold_answer": "Dự kiến đón khoảng 6 triệu lượt khách, trong đó khách quốc tế khoảng 1.400.000 lượt vào năm 2030."
    }
]

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
            keywords = []
            if q_id == 1: keywords = ["PLX", "0,39%"]
            elif q_id == 2: keywords = ["45 triệu", "3 triệu"]
            elif q_id == 3: keywords = ["15%", "báo in"]
            elif q_id == 4: keywords = ["2024"]
            elif q_id == 5: keywords = ["6 triệu", "1.400.000"]
            
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
