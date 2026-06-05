import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

from src.store import EmbeddingStore
from src.embeddings import OpenAIEmbedder

# Nạp cấu hình từ file .env
load_dotenv(override=False)

# Phím từ khóa kiểm định tương ứng với 10 câu hỏi để tự động xác minh kết quả Top-3
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
    if not Path(file_path).exists():
        print(f"Lỗi: Không tìm thấy file {file_path}!")
        sys.exit(1)
        
    content = Path(file_path).read_text(encoding="utf-8")
    
    # Sử dụng biểu thức chính quy để bóc tách thông tin
    questions = re.findall(r"Câu hỏi (\d+): (.*?)\nTài liệu đích: (.*?)\nCâu trả lời đúng: (.*?)\n", content)
    
    qa_list = []
    for q_id_str, query, source, gold in questions:
        q_id = int(q_id_str)
        qa_list.append({
            "id": q_id,
            "query": query.strip(),
            "source": source.strip(),
            "gold_answer": gold.strip(),
            "keywords": KEYWORDS_MAP.get(q_id, [])
        })
    return qa_list

def main():
    print("=== CHƯƠNG TRÌNH KIỂM THỬ VÀ ĐÁNH GIÁ CHẤT LƯỢNG RETRIEVAL ===")
    
    # 1. Kiểm tra API Key và khởi tạo OpenAI embedder
    try:
        embedder = OpenAIEmbedder()
        print(f"[*] Khởi tạo thành công OpenAI Embedder ({embedder.model_name})")
    except Exception as e:
        print(f"Lỗi khởi tạo OpenAI API: {e}")
        sys.exit(1)
        
    # 2. Kiểm tra các collection tồn tại trong ChromaDB
    persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    if not persist_dir:
        print("Lưu ý: CHROMA_PERSIST_DIR chưa được bật trong file .env.")
        print("Mặc định chương trình sẽ kiểm thử ở chế độ in-memory trống.")
        
    print("\nChọn bộ dữ liệu (collection) bạn muốn kiểm thử:")
    print("1. interactive_store_600 (Kích thước chunk 600)")
    print("2. interactive_store_1200 (Kích thước chunk 1200)")
    choice = input("Lựa chọn của bạn (1 hoặc 2): ").strip()
    chunk_size = 1200 if choice == "2" else 600
    collection_name = f"interactive_store_{chunk_size}"
    
    # 3. Khởi tạo kết nối Vector Store
    store = EmbeddingStore(collection_name=collection_name, embedding_fn=embedder)
    size = store.get_collection_size()
    print(f"\n[*] Đang kết nối tới Collection: '{collection_name}' (ChromaDB)")
    print(f"[*] Tổng số lượng chunks đang lưu trữ: {size}")
    
    if size == 0:
        print("Cảnh báo: Bộ lưu trữ hiện đang trống rỗng (0 chunks)!")
        print("Vui lòng chạy file chat_agent.py trước để nạp dữ liệu và tạo embeddings.")
        sys.exit(1)
        
    # 4. Phân tích bộ benchmark_qa.txt
    qa_pairs = parse_benchmark_qa()
    print(f"[*] Đã tải thành công {len(qa_pairs)} câu hỏi từ data/benchmark_qa.txt\n")
    
    print("=" * 80)
    print(f"{'Q#':<3} | {'Trạng thái':<12} | {'Score Max':<10} | {'Tiêu đề bài báo liên quan nhất'}")
    print("=" * 80)
    
    passed_count = 0
    results_detail = []
    
    for item in qa_pairs:
        q_id = item["id"]
        query = item["query"]
        keywords = item["keywords"]
        
        # Tìm kiếm ngữ nghĩa
        search_results = store.search(query, top_k=3)
        
        found_in_top_3 = False
        top_score = 0.0
        best_title = "N/A"
        
        if search_results:
            top_score = search_results[0]["score"]
            best_title = search_results[0]["metadata"].get("title", "Không rõ")
            
            # Kiểm tra xem từ khóa chuẩn có nằm trong các chunk thuộc top 3 hay không
            for res in search_results:
                chunk_content = res["content"].lower()
                # Kiểm tra nếu khớp tất cả từ khóa trong danh sách keywords
                if all(kw.lower() in chunk_content for kw in keywords):
                    found_in_top_3 = True
                    break
                    
        status = "PASSED ✅" if found_in_top_3 else "FAILED ❌"
        if found_in_top_3:
            passed_count += 1
            
        print(f"{q_id:<3} | {status:<12} | {top_score:<10.4f} | {best_title}")
        results_detail.append((q_id, query, status, top_score, best_title, search_results))
        
    accuracy = (passed_count / len(qa_pairs)) * 100
    print("=" * 80)
    print(f"TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ CHẤT LƯỢNG:")
    print(f"- Số câu hỏi vượt qua (Top-3 chứa ngữ cảnh đúng): {passed_count}/{len(qa_pairs)}")
    print(f"- Tỷ lệ chính xác retrieval (Accuracy): {accuracy:.1f}%")
    print("=" * 80)
    
    # 5. In chi tiết các câu bị thất bại để người dùng tự phân tích
    if passed_count < len(qa_pairs):
        print("\n=== CHI TIẾT CÁC CÂU RETRIEVAL THẤT BẠI (FAILED) ===")
        for q_id, query, status, score, title, res_list in results_detail:
            if status == "FAILED ❌":
                print(f"\n[Q{q_id}] {query}")
                print(f"  -> Điểm cao nhất: {score:.4f} | Bài báo: {title}")
                print("  -> Top 3 chunks trả về không chứa đầy đủ thông tin đáp án.")
                if res_list:
                    print("     [Xem trước nội dung Chunk 1]:")
                    print(f"     \"{res_list[0]['content'][:200]}...\"")
                print("-" * 50)

if __name__ == "__main__":
    main()
