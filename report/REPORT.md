# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Lê Thanh Điệp
**Nhóm:** Nhóm A2
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity (điểm số gần bằng 1.0) nghĩa là hai vector đại diện cho hai văn bản hướng về cùng một phía trong không gian đa chiều ngữ nghĩa. Về mặt thực tế, điều này chỉ ra hai văn bản có sự tương đồng rất lớn về mặt ngữ cảnh, chủ đề hoặc từ vựng, độc lập với độ dài ký tự của chúng.

**Ví dụ HIGH similarity:**
- Sentence A: "Lập trình Python rất được ưa chuộng nhờ cú pháp dễ đọc và thư viện phong phú."
- Sentence B: "Python là một ngôn ngữ lập trình tuyệt vời vì nó đơn giản và sở hữu hệ sinh thái mạnh mẽ."
- Tại sao tương đồng: Cả hai câu đều chia sẻ chung một luận điểm cốt lõi là ca ngợi các ưu điểm của ngôn ngữ Python (dễ đọc/đơn giản, thư viện phong phú/hệ sinh thái mạnh mẽ).

**Ví dụ LOW similarity:**
- Sentence A: "Chỉ số VN-Index hôm nay bốc hơi hơn 15 điểm do lực bán tháo cổ phiếu blue-chips."
- Sentence B: "Để làm món phở bò truyền thống, bạn cần hầm xương ống tối thiểu trong vòng 8 tiếng."
- Tại sao khác: Hai câu nói về hai lĩnh vực hoàn toàn xa lạ (thị trường tài chính chứng khoán và công thức nấu ăn ẩm thực).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Euclidean distance đo khoảng cách thẳng giữa hai điểm cuối vector nên cực kỳ nhạy cảm với độ dài văn bản (văn bản dài chứa nhiều từ hơn sẽ có độ dài vector lớn hơn, kéo chúng ra xa nhau). Trái lại, Cosine similarity chỉ đo góc giữa hai vector (hướng ngữ nghĩa), loại bỏ hoàn toàn yếu tố độ dài văn bản, giúp so sánh công bằng giữa đoạn văn ngắn và bài viết dài.

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

**Domain:** Tin tức Kinh tế - Tài chính Việt Nam (VnEconomy & Báo Đầu tư).

**Tại sao nhóm chọn domain này?**
> Tài liệu tin tức kinh tế chứa lượng thông tin số liệu dày đặc (phần trăm biến động, lượng tiền đầu tư, mốc thời gian và các luật định) rất nhạy cảm với độ chính xác. Đây là môi trường hoàn hảo để thử nghiệm khả năng định vị thông tin chuẩn xác của các chiến lược RAG, đồng thời đánh giá hiệu quả lọc dữ liệu dựa trên metadata như thể loại bài báo hay thời gian phát hành.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Hàng về tranh bán, thị trường chìm trong sắc đỏ... | vneconomy.vn | 1,840 | `{"category": "Chứng khoán", "time": "2023-03-03"}` |
| 2 | Bộ Tài chính lấy ý kiến đề xuất ưu đãi thuế TN doanh nghiệp | vneconomy.vn | 1,780 | `{"category": "Tài chính", "time": "2024-06-13"}` |
| 3 | Panasonic khai trương nhà máy thiết bị chất lượng không khí | baodautu.vn | 1,410 | `{"category": "Doanh nghiệp", "time": "2021-09-30"}` |
| 4 | “Dư âm” từ đại dịch kéo dài, các hãng hàng không phục hồi... | vneconomy.vn | 4,520 | `{"category": "Đầu tư", "time": "2022-09-19"}` |
| 5 | Ven biển Ninh Thuận trở thành vùng du lịch trọng điểm | vneconomy.vn | 1,120 | `{"category": "Bất động sản", "time": "2022-02-25"}` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `category` | string | `"Chứng khoán"` | Dùng để pre-filter (lọc trước) các chủ đề không liên quan, tăng tốc độ và độ chính xác của RAG. |
| `time` | string | `"2023-03-03"` | Hỗ trợ truy vấn thông tin theo mốc thời gian cụ thể (ví dụ: tin tức năm 2023 hoặc 2024). |
| `doc_id` | string | `"doc_0"` | Quản lý vòng đời tài liệu, phục vụ chức năng xóa toàn bộ các chunks thuộc về một tài liệu gốc. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên các tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Panasonic | FixedSizeChunker (`fixed_size`) | 4 | 352 | Trung bình (cắt ngang từ tại điểm cuối của chunk) |
| Panasonic | SentenceChunker (`by_sentences`) | 2 | 705 | Tốt (ngắt câu hoàn chỉnh, giữ nguyên ngữ cảnh câu) |
| Panasonic | RecursiveChunker (`recursive`) | 3 | 470 | Rất tốt (cắt theo dấu xuống dòng kép và phân đoạn tự nhiên) |

### Strategy Của Tôi

**Loại:** Recursive Character Splitting (`RecursiveChunker`).

**Mô tả cách hoạt động:**
> Chunker hoạt động dựa trên thuật toán đệ quy tìm kiếm các ký tự ngắt dòng theo độ ưu tiên giảm dần: `["\n\n", "\n", ". ", " ", ""]`. Nếu một đoạn văn vượt quá kích thước `chunk_size` yêu cầu, nó sẽ cắt tại dấu phân tách cao nhất (ngắt đoạn `\n\n`), sau đó đệ quy sâu xuống các phần nhỏ để cắt theo dòng `\n`, câu `. ` hoặc khoảng trắng `" "` cho đến khi tất cả các mảnh đều thỏa mãn giới hạn kích thước.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Báo chí tài chính có cấu trúc phân đoạn rõ ràng bằng cách xuống dòng kép `\n\n`. Mỗi đoạn văn thường chứa trọn vẹn một thông tin hoặc một bảng số liệu độc lập. Recursive chunking tôn trọng cấu trúc tự nhiên này, giúp các thông tin ngữ nghĩa được giữ chung với nhau tốt hơn so với cắt cứng theo số ký tự.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Panasonic | Best Baseline (`by_sentences`) | 2 | 705 | Tốt (truy xuất được câu trọn vẹn) |
| Panasonic | **của tôi** (`recursive` - 600) | 3 | 470 | Rất tốt (định vị đúng đoạn chứa thông tin kinh tế) |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Recursive (600) | 6.0/10 | Ngắt đoạn tự nhiên, giữ cấu trúc nguyên bản | Không tự gộp các đoạn quá ngắn, gây phân mảnh số liệu |
| Nguyễn Văn A | Fixed Size (500) | 4.0/10 | Số lượng chunk và độ dài đồng đều | Phá vỡ câu từ ở ranh giới, gây lỗi tìm kiếm |
| Trần Thị B | Sentence (max 3) | 8.0/10 | Giữ câu hoàn chỉnh, ngữ cảnh liền mạch | Kích thước các chunk biến động mạnh không đều |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Chiến lược **Sentence-based** kết hợp gộp đoạn (Merging) là tốt nhất. Đối với tin tức kinh tế nhiều số liệu, việc giữ các câu nguyên vẹn và liên kết chúng lại trong một cửa sổ trượt đảm bảo các số liệu bổ trợ (ví dụ: tên công ty + số tiền đầu tư) không bị tách sang 2 chunk riêng biệt.

---

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
| 1 | Vào phiên giao dịch ngày 03/03/2023, chỉ số VN-Index đóng cửa giảm bao nhiêu phần trăm và rổ VN30 có duy nhất mã cổ phiếu nào tăng giá? | VN-Index đóng cửa giảm 1,24% và mã cổ phiếu tăng giá duy nhất trong rổ VN30 là PLX (tăng 0,39%). |
| 2 | Panasonic đã đầu tư bao nhiêu USD vào nhà máy mới tại Bình Dương và năng suất dự kiến vào năm 2025 là bao nhiêu? | Vốn đầu tư khoảng 45 triệu USD và năng suất dự kiến đạt khoảng 3 triệu sản phẩm vào năm 2025. |
| 3 | Bộ Tài chính đề xuất ưu đãi thuế thu nhập doanh nghiệp bao nhiêu phần trăm đối với cơ quan báo chí ngoài báo in? | Bộ Tài chính đề xuất thuế suất ưu đãi 15% đối với thu nhập của các cơ quan báo chí ngoài báo in. |
| 4 | Theo dự báo của IATA, thị trường hàng không thế giới sẽ phục hồi vượt mức trước dịch vào năm nào? | IATA dự báo thị trường hàng không thế giới sẽ phục hồi vượt mức trước dịch Covid-19 vào đầu năm 2024. |
| 5 | Khu du lịch quốc gia Ninh Chữ dự kiến đón bao nhiêu lượt khách du lịch và bao nhiêu khách quốc tế vào năm 2030? | Dự kiến đón khoảng 6 triệu lượt khách, trong đó khách quốc tế khoảng 1.400.000 lượt vào năm 2030. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Vào phiên 03/03/2023... | "Các chỉ số đều thể hiện một nhịp phục hồi khá nhanh..." | 0.6987 | Một phần | VN-Index đóng cửa giảm 1,24% nhưng không lấy được mã PLX. |
| 2 | Panasonic Bình Dương... | "Nhà máy mới được xây dựng tại tỉnh Bình Dương..." | 0.6158 | Một phần | Panasonic đầu tư 45 triệu USD nhưng thiếu năng suất 2025. |
| 3 | Ưu đãi thuế báo chí... | "Đáng chú ý, dự thảo bổ sung áp thuế suất ưu đãi 15%..." | 0.7405 | Có | Đề xuất áp dụng mức thuế suất ưu đãi 15% ngoài báo in. |
| 4 | Dự báo phục hồi IATA... | "IATA dự báo hàng không sẽ phục hồi vượt mức..." | 0.7119 | Có | Thị trường hàng không sẽ phục hồi vào đầu năm 2024. |
| 5 | Khách du lịch Ninh Chữ...| "Đón khoảng 6 triệu lượt khách (quốc tế 1.400.000)..." | 0.7780 | Có | Đón 6 triệu khách và 1.400.000 khách quốc tế năm 2030. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 3 / 5

*Giải thích lý do Q1 và Q2 bị lỗi:* Do thuật toán ngắt dòng đệ quy ngắt theo các đoạn văn `\n\n` tự nhiên. Khi mỗi đoạn văn ngắn hơn kích thước chunk giới hạn, chúng được lưu trữ độc lập mà không có bước gộp lại. Vì thông tin câu trả lời chuẩn của Q1 và Q2 phân mảnh nằm ở 2 đoạn văn liền kề nhau, kết quả trả về không thể chứa đồng thời cả hai thông tin này trong một chunk đơn lẻ.

---

## 7. What I Learned (5 điểm — Demo)

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
