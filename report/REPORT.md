# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Lê Thanh Điệp
**Nhóm:** Nhóm A2
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector đại diện cho hai văn bản hướng về cùng một phía trong không gian đa chiều ngữ nghĩa. Vì hai văn bản có sự tương đồng rất lớn về mặt ngữ cảnh, chủ đề hoặc từ vựng.

**Ví dụ HIGH similarity:**
- Sentence A: "Lập trình Python rất được ưa chuộng nhờ cú pháp dễ đọc và thư viện phong phú."
- Sentence B: "Python là một ngôn ngữ lập trình tuyệt vời vì nó đơn giản và sở hữu hệ sinh thái mạnh mẽ."
- Tại sao tương đồng: Cả hai câu đều chia sẻ một luận điểm cốt lõi là ca ngợi các ưu điểm của ngôn ngữ Python (dễ đọc/đơn giản, thư viện phong phú/hệ sinh thái mạnh mẽ).

**Ví dụ LOW similarity:**
- Sentence A: "Chỉ số VN-Index hôm nay bốc hơi hơn 15 điểm do lực bán tháo cổ phiếu blue-chips."
- Sentence B: "Để làm món phở bò truyền thống, bạn cần hầm xương ống tối thiểu trong vòng 8 tiếng."
- Tại sao khác: Hai câu nói về hai lĩnh vực khác nhau, nên similarity sẽ rất thấp

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Euclidean distance đo khoảng cách thẳng giữa hai điểm cuối vector nên cực kỳ nhạy cảm với độ dài văn bản (văn bản dài chứa nhiều từ hơn sẽ có độ dài vector lớn hơn, kéo chúng ra xa nhau). Trái lại, Cosine similarity chỉ đo góc giữa hai vector (hướng ngữ nghĩa), loại bỏ yếu tố độ dài văn bản, giúp so sánh công bằng giữa đoạn văn ngắn và bài viết dài.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> Công thức tính số lượng chunk: 
> $$num\_chunks = \lceil \frac{doc\_length - overlap}{chunk\_size - overlap} \rceil$$
> Áp dụng số liệu:
> $$num\_chunks = \lceil \frac{10000 - 50}{500 - 50} \rceil = \lceil \frac{9950}{450} \rceil = \lceil 22.11 \rceil = 23$$
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Trình bày phép tính:*
> $$num\_chunks = \lceil \frac{10000 - 100}{500 - 100} \rceil = \lceil \frac{9900}{400} \rceil = \lceil 24.75 \rceil = 25$$ chunks.
> Số lượng chunk tăng từ 23 lên 25.
> *Tại sao muốn overlap nhiều hơn:* Việc tăng overlap giúp bảo toàn các ngữ cảnh quan trọng nằm ở vùng ranh giới giữa hai chunk, ngăn ngừa việc các câu văn hay số liệu đi liền nhau bị cắt đôi làm mất ý nghĩa khi đưa vào LLM.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Tin tức Kinh tế - Tài chính Việt Nam.

**Tại sao nhóm chọn domain này?**
> Tài liệu tin tức kinh tế chứa lượng thông tin số liệu dày đặc (phần trăm biến động, lượng tiền đầu tư, mốc thời gian và các luật định) rất nhạy cảm với độ chính xác. Đây là môi trường hoàn hảo để thử nghiệm khả năng định vị thông tin chuẩn xác của các chiến lược RAG, đồng thời đánh giá hiệu quả lọc dữ liệu dựa trên metadata như thể loại bài báo hay thời gian phát hành.

### Data Inventory (Mẫu 5 tài liệu tiêu biểu trong bộ dữ liệu)

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Cổ phiếu VGC của Viglacera tăng nhẹ phiên chào sàn HNX | baodautu.vn | 2,999 | `{"category": "Doanh nghiệp", "time": "2016-12-22"}` |
| 2 | Khánh Hòa rà soát lại tiến độ thực hiện các dự án trên địa bàn tỉnh | vneconomy.vn | 2,777 | `{"category": "Bất động sản", "time": "2021-12-15"}` |
| 3 | Vàng có thể phá ngưỡng hỗ trợ 1.800 USD/ounce | baodautu.vn | 2,782 | `{"category": "Ngân hàng - Bảo hiểm", "time": "2020-11-27"}` |
| 4 | Thúc tiến độ lắp camera trên xe kinh doanh vận tải... | vneconomy.vn | 2,744 | `{"category": "Đầu tư", "time": "2021-09-15"}` |
| 5 | Đầu tư hạ tầng giao thông Đèo Cả đăng ký niêm yết trên HOSE | baodautu.vn | 2,859 | `{"category": "Chứng khoán", "time": "2021-05-28"}` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `category` | string | `"Chứng khoán"` | Dùng để pre-filter (lọc trước) các chủ đề không liên quan, tăng tốc độ và độ chính xác của RAG. |
| `time` | string | `"2023-03-03"` | Hỗ trợ truy vấn thông tin theo mốc thời gian cụ thể (ví dụ: tin tức năm 2023 hoặc 2024). |
| `doc_id` | string | `"doc_0"` | Quản lý vòng đời tài liệu, phục vụ chức năng xóa toàn bộ các chunks thuộc về một tài liệu gốc. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên tài liệu đầu tiên (**Viglacera**):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Viglacera | FixedSizeChunker (`fixed_size` - 600) | 6 | 541.50 | Trung bình (cắt ngang từ tại điểm cuối của chunk) |
| Viglacera | SentenceChunker (`by_sentences`) | 7 | 427.57 | Tốt (ngắt câu hoàn chỉnh, giữ nguyên ngữ cảnh câu) |
| Viglacera | RecursiveChunker (`recursive` - 600) | 13 | 229.77 | Rất tốt (cắt theo dấu xuống dòng kép và phân đoạn tự nhiên) |

### Strategy Của Tôi

**Loại:** Recursive Character Splitting (`RecursiveChunker`).

**Mô tả cách hoạt động:**
> Chunker hoạt động dựa trên thuật toán đệ quy tìm kiếm các ký tự ngắt dòng theo độ ưu tiên giảm dần: `["\n\n", "\n", ". ", " ", ""]`. Nếu một đoạn văn vượt quá kích thước `chunk_size` yêu cầu, nó sẽ cắt tại dấu phân tách cao nhất (ngắt đoạn `\n\n`), sau đó đệ quy sâu xuống các phần nhỏ để cắt theo dòng `\n`, câu `. ` hoặc khoảng trắng `" "` cho đến khi tất cả các mảnh đều thỏa mãn giới hạn kích thước.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Báo chí tài chính có cấu trúc phân đoạn rõ ràng bằng cách xuống dòng kép `\n\n`. Mỗi đoạn văn thường chứa trọn vẹn một thông tin hoặc một bảng số liệu độc lập. Recursive chunking tôn trọng cấu trúc tự nhiên này, giúp các thông tin ngữ nghĩa được giữ chung với nhau tốt hơn so với cắt cứng theo số ký tự.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|-------------------|
| Viglacera | Best Baseline (`by_sentences`) | 7 | 427.57 | Tốt (truy xuất được câu trọn vẹn) |
| Viglacera | **của tôi** (`recursive` - 600) | 13 | 229.77 | Rất tốt (định vị đúng đoạn chứa thông tin kinh tế) |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi (Thanh Điệp) | Recursive Character Splitting (600) | 7.0/10 | Ngắt đoạn tự nhiên theo `\n\n` hoặc câu, giữ cấu trúc bài báo tốt. | Không tự gộp các đoạn quá ngắn, dễ bị ngắt rời từ khóa kiểm thử (như câu 4). |
| Đỗ Minh Phúc | Semantic Chunking | 9.0/10 | Chia chunk dựa trên ranh giới ngữ nghĩa của các câu liên tiếp, giữ trọn vẹn ngữ cảnh. | Chi phí tính toán embedding khi phân chia chunk lớn và phức tạp. |
| Phí Đình Mạnh | Document-structure Chunking | 7.0/10 | Phân chia theo cấu trúc phân tầng (đề mục, Markdown) tự nhiên. | Kém hiệu quả trên tài liệu phẳng ít cấu trúc (như bảng tin tức csv). |
| Lê Anh Minh | Hybrid/Semantic  | 10.0/10 (Recall@3: 100%) | Không gian tìm kiếm tối giản, kết hợp lai giúp định vị tối đa. | Chưa thử nghiệm trên tập dữ liệu rộng (100 tài liệu) nên chưa phản ánh hết nhiễu. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Chiến lược **Sentence-based** kết hợp gộp đoạn (Merging) là tốt nhất. Đối với tin tức kinh tế nhiều số liệu, việc giữ các câu nguyên vẹn và liên kết chúng lại trong một cửa sổ trượt đảm bảo các số liệu bổ trợ (ví dụ: tên công ty + số tiền đầu tư) không bị tách sang 2 chunk riêng biệt.

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng biểu thức chính quy `re.split(r'(?<=[.!?])\s+', text)` để tìm kiếm ranh giới câu dựa trên dấu chấm/hỏi/cảm thán theo sau bởi khoảng trắng. Các câu sau khi tách được dùng hàm `.strip()` loại bỏ khoảng trắng thừa, loại bỏ câu rỗng và gom nhóm tối đa theo số lượng `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thiết kế thuật toán đệ quy. Base case của đệ quy là khi văn bản đầu vào có độ dài nhỏ hơn `chunk_size` hoặc danh sách ký tự ngăn cách cạn kiệt, hàm sẽ trả về văn bản đó ngay lập tức. Ngược lại, nó thực hiện tách chuỗi bằng ký tự ngăn cách đầu tiên trong danh sách và gọi đệ quy tiếp tục trên từng mảnh.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Cơ sở dữ liệu in-memory lưu trữ danh sách các bản ghi dạng dictionary có đầy đủ `id`, `content`, `metadata` và `embedding` được sinh ra bởi `OpenAIEmbedder`. Hàm tìm kiếm `search` tính toán Cosine Similarity bằng hàm `compute_similarity` trên toàn bộ store, sắp xếp giảm dần theo điểm tương đồng (`score`) và trả về `top_k` kết quả.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` áp dụng cơ chế lọc trước (pre-filtering) bằng cách kiểm tra xem dictionary metadata của từng chunk có chứa đầy đủ cặp khóa-giá trị của `metadata_filter` hay không, sau đó mới chạy so khớp vector trên các chunk vượt qua bộ lọc. `delete_document` thực hiện lọc bỏ các bản ghi trong danh sách store cục bộ có chứa `metadata['doc_id'] == doc_id`.

### KnowledgeBaseAgent

**`answer`** — approach:
> Truy vấn `top_k` các chunk liên quan nhất từ store, trích xuất chuỗi nội dung văn bản (`content`) của chúng, ghép nối thành một chuỗi ngữ cảnh thống nhất phân tách bằng `\n---\n`. Cuối cùng, bọc ngữ cảnh này cùng câu hỏi của người dùng vào prompt mẫu và đẩy sang hàm LLM để lấy câu trả lời.

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.07s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Thị trường chứng khoán hôm nay đỏ sàn. | Chỉ số VN-Index bốc hơi hơn 15 điểm vào phiên chiều. | High | 0.7866 | Đúng |
| 2 | Panasonic mở nhà máy sản xuất quạt trần tại Bình Dương. | Hãng điện tử Nhật Bản xây dựng cơ sở sản xuất thiết bị IAQ. | High | 0.7205 | Đúng |
| 3 | Bộ Tài chính lấy ý kiến về ưu đãi thuế thu nhập doanh nghiệp. | Dự thảo điều chỉnh mức giảm thuế thu nhập cho các tổ chức kinh tế. | High | 0.8115 | Đúng |
| 4 | Học máy sử dụng dữ liệu để cải thiện thuật toán. | Thời tiết hôm nay mát mẻ và có gió nhẹ. | Low | 0.1240 | Đúng |
| 5 | Ngành hàng không có dấu hiệu phục hồi tích cực sau dịch. | Tàu bay Vietnam Airlines tăng thêm tần suất đi Seoul. | High | 0.6890 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là Cặp số 2 đạt điểm số khá cao (~0.72) mặc dù từ vựng phân bổ giữa hai câu rất khác nhau (một câu ghi "quạt trần tại Bình Dương", câu kia ghi "thiết bị IAQ"). Điều này chứng tỏ embeddings biểu diễn nghĩa ở cấp độ khái niệm (concept) và ngữ cảnh chứ không đơn thuần chỉ là so khớp từ khóa (keyword matching).

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Cổ phiếu VGC của Viglacera niêm yết trên sàn giao dịch nào? | Cổ phiếu VGC chính thức niêm yết trên sàn HNX (Sở Giao dịch Chứng khoán Hà Nội). |
| 2 | Tỉnh Khánh Hòa rà soát lại tiền thực hiện các dự án nào? | Khánh Hòa rà soát tiến độ 29 dự án, phát hiện 13 dự án đang tạm dừng triển khai do vướng mắc về quy hoạch; hoàn thành 3 dự án và thu hồi/hủy thông báo cho phép đầu tư đối với 4 dự án (Khu biệt thự và du lịch Đồng Bé, Làng biệt thự Tâm Hương, Khu vườn tượng đá nghệ thuật Nha Trang, Bệnh viện đa khoa quốc tế Nha Trang). |
| 3 | Goldman Sachs dự báo giá vàng đạt bao nhiêu USD mỗi ounce? | Goldman Sachs dự báo giá vàng năm 2021 sẽ đạt mức 2.300 USD/ounce. |
| 4 | Lợi ích của việc lắp camera giám sát trên xe kinh doanh vận tải là gì? | Lợi ích là nâng cao hiệu quả quản lý hành trình của đơn vị kinh doanh vận tải, bảo đảm trật tự an toàn giao thông và hỗ trợ đắc lực cho công tác điều tra, giải quyết tai nạn của cơ quan Công an. |
| 5 | Đường cao tốc nào được khánh thành kỷ niệm ngày truyền thống GTVT? | Dự án hầm đường bộ qua đèo Cả / đường cao tốc hạ tầng giao thông Đèo Cả (HHV). |
| 6 | Giá trị giao dịch bất động sản khu vực châu Thái Bình Dương là bao nhiêu tỷ USD? | Đạt 83,5 tỷ USD trong 6 tháng đầu năm 2021 (tăng 39% so với cùng kỳ năm 2020). |
| 7 | MBV ra mắt dịch vụ hay sản phẩm mới nào cho khách hàng? | MBV ra mắt diện mạo nhận diện thương hiệu mới cùng cam kết chuyển đổi số để mang lại những trải nghiệm tài chính số ưu việt cho khách hàng. |
| 8 | Thị trường bất động sản Hà Nội phân khúc nào đang nóng nhất? | Phân khúc biệt thự và nhà ở liền kề có vị trí tốt, giao thông thuận lợi, nằm trong khuôn viên xanh - sạch - đẹp. |
| 9 | Lũ quét gây sự cố trên công trình thủy điện xanh tại tỉnh nào? | Xảy ra tại tỉnh Quảng Ngãi (Công trình thủy điện Đăk Re). |
| 10 | Bộ Thông tin truyền thông đề xuất giải pháp nào để giải quyết tình trạng SIM rác? | Đề xuất thực hiện kết nối thông tin thuê bao với Cơ sở dữ liệu quốc gia về dân cư nhằm đối soát, xác thực chuẩn thông tin thuê bao. |

### Kết Quả Của Tôi (Chạy bằng `run_benchmark_eval.py`)

| # | Query | Trạng thái | Score Max | Tiêu đề bài báo liên quan nhất (Top-1) | Relevant? |
|---|-------|------------|-----------|----------------------------------------|-----------|
| 1 | Cổ phiếu VGC của Viglacera niêm yết trên sàn giao dịch nào? | PASSED ✅ | 0.4621 | Cổ phiếu VGC của Viglacera tăng nhẹ phiên chào sàn HNX | Có |
| 2 | Tỉnh Khánh Hòa rà soát lại tiền thực hiện các dự án nào? | PASSED ✅ | 0.3231 | Khánh Hòa rà soát lại tiến độ thực hiện các dự án trên địa bàn tỉnh | Có |
| 3 | Goldman Sachs dự báo giá vàng đạt bao nhiêu USD mỗi ounce? | PASSED ✅ | 0.3996 | Vàng có thể phá ngưỡng hỗ trợ 1.800 USD/ounce | Có |
| 4 | Lợi ích của việc lắp camera giám sát trên xe kinh doanh vận tải là gì? | FAILED ❌ | 0.5742 | Thúc tiến độ lắp camera trên xe kinh doanh vận tải, hoàn thành trước ngày 31/12/2021 | Có |
| 5 | Đường cao tốc nào được khánh thành kỷ niệm ngày truyền thống GTVT? | FAILED ❌ | -0.2142 | Đầu tư gần 5.390 tỷ đồng cho tuyến đường song hành Vành đai 4 - Vùng Thủ đô | Không |
| 6 | Giá trị giao dịch bất động sản khu vực châu Thái Bình Dương là bao nhiêu tỷ USD? | PASSED ✅ | 0.0550 | Giá trị giao dịch bất động sản châu Á -Thái Bình Dương 6 tháng đạt 83,5 tỷ USD | Có |
| 7 | MBV ra mắt dịch vụ hay sản phẩm mới nào cho khách hàng? | PASSED ✅ | 0.1951 | MBV ra mắt diện mạo mới: Dấu ấn hiện đại và kết nối | Có |
| 8 | Thị trường bất động sản Hà Nội phân khúc nào đang nóng nhất? | PASSED ✅ | 0.1715 | Thị trường bất động sản thấp tầng Hà Nội: Lăng kính từ một dự án | Có |
| 9 | Lũ quét gây sự cố trên công trình thủy điện xanh tại tỉnh nào? | FAILED ❌ | 0.0623 | 3 tỉnh, thành phố đã cơ bản được cấp điện trở lại | Không |
| 10 | Bộ Thông tin truyền thông đề xuất giải pháp nào để giải quyết tình trạng SIM rác? | PASSED ✅ | 0.2563 | Giải quyết dứt điểm tình trạng sử dụng SIM rác, SIM nặc danh | Có |

**TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ CHẤT LƯỢNG:**
- **Số câu hỏi vượt qua (Top-3 chứa ngữ cảnh đúng và đủ từ khóa):** 7/10
- **Tỷ lệ chính xác retrieval (Accuracy):** 70.0%

---

## 7. What I Learned (5 điểm — Demo)

### Phân tích nguyên nhân lỗi (Failure Analysis)

Từ 3 trường hợp thất bại (FAILED ❌) ở trên, chúng ta rút ra các nguyên nhân chính sau:

1. **Lỗi xác thực từ khóa kiểm tra (Query 4 - Lắp camera giám sát):**
   - *Triệu chứng:* RAG hệ thống đã tìm được chunk đầu tiên hoàn toàn chính xác từ bài báo mục tiêu ("Thúc tiến độ lắp camera trên xe kinh doanh vận tải...") với score rất cao (`0.5742`). Tuy nhiên, câu này bị đánh giá là **FAILED ❌**.
   - *Nguyên nhân:* Bộ kiểm định `KEYWORDS_MAP` định nghĩa từ khóa là `["hành trình", "an toàn"]` dựa trên câu trả lời chuẩn (Gold Answer). Nhưng thực tế trong văn bản gốc bài báo, câu nói về lợi ích chỉ ghi là *"...giám sát được trạng thái của lái xe, như việc lái xe nghe điện thoại, mất tập trung và các hành vi gây mất an toàn giao thông khác"* mà không chứa từ *"hành trình"*. Do đó, việc so khớp từ khóa bị fail mặc dù retrieval đã đúng chunk liên quan.
   - *Đề xuất cải thiện:* Cần tinh chỉnh bộ từ khóa kiểm thử (ground truth keywords) bám sát văn bản gốc thay vì bám hoàn toàn vào câu trả lời tự viết của con người.

2. **Lỗi thiếu thông tin trong tài liệu nguồn (Query 5 & Query 9):**
   - *Triệu chứng:* Hệ thống trả về các văn bản hoàn toàn không liên quan (ví dụ: truy vấn cao tốc đèo Cả lại ra Vành đai 4 / EVN; truy vấn thủy điện xanh lại ra thông tin khắc phục lưới điện sau bão).
   - *Nguyên nhân:* Khi kiểm tra kỹ nội dung của bài báo gốc `doc_4` ("Đầu tư hạ tầng giao thông Đèo Cả đăng ký niêm yết trên HOSE") và `doc_8` ("Nước rút trên công trình thủy điện 'xanh' tại Quảng Ngãi") trong cơ sở dữ liệu `100_data.csv`, ta nhận thấy:
     - Bài báo Đèo Cả hoàn toàn nói về việc đăng ký niêm yết cổ phiếu HHV lên sàn HOSE, kế hoạch tăng vốn và doanh thu thu phí, không hề có cụm từ hay thông tin nào nói về việc *"khánh thành kỷ niệm ngày truyền thống GTVT"*.
     - Bài báo Thủy điện Đăk Re chỉ tập trung nói về biện pháp thi công đào hầm tối ưu để giữ rừng và không phải di dời dân, hoàn toàn không có thông tin nào về việc *"Lũ quét gây sự cố"*.
   - Do đó, đây là các câu hỏi nằm ngoài phạm vi thông tin có sẵn trong tài liệu nguồn (Out-of-document queries). Embeddings mô hình không thể tìm thấy sự tương đồng ngữ nghĩa thực tế, dẫn đến việc lấy các chunk khác có độ tương đồng nhiễu lớn nhất nhưng điểm số cực kỳ thấp (thậm chí âm `-0.2142`).
   - *Đề xuất cải thiện:* Bổ sung dữ liệu nguồn đầy đủ hoặc xây dựng cơ chế phát hiện câu hỏi không có câu trả lời (Unanswerable query detection) khi mức độ tương đồng tối đa nằm dưới một ngưỡng cut-off (ví dụ: < 0.15).

### Bài học kinh nghiệm

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được rằng việc tăng kích thước `overlap` trong chiến lược Sentence-based đóng vai trò cực kỳ quan trọng đối với các câu hỏi so sánh số liệu hoặc cần liên kết thông tin giữa các câu văn liên tiếp.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nhóm bạn đã trình bày giải pháp tự động sinh và đánh giá câu hỏi benchmark bằng LLM (LLM-as-a-judge), điều này giúp tiết kiệm rất nhiều công sức so với việc nhóm tự biên soạn và đánh giá kết quả thủ công.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi chắc chắn sẽ cài đặt thêm cơ chế **Gộp Chunks (Chunk Merging Window)** cho thuật toán phân tách đệ quy để đảm bảo các đoạn văn ngắn nằm cạnh nhau sẽ được ghép lại làm tăng tính bao quát thông tin của ngữ cảnh khi đưa vào RAG.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
