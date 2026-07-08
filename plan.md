# KẾ HOẠCH TRIỂN KHAI: SỐ HÓA VÀ TÁI CẤU TRÚC FLOWCHART

Tài liệu này mô tả thiết kế triển khai của hệ thống **FlowRevamp** — số hóa flowchart từ ảnh bằng Python + OpenCV, không dùng Deep Learning (ngoại trừ EasyOCR, có thể tắt). Hệ thống được cấu trúc thành pipeline **4 bước tuần tự**, do một bộ điều phối (**Orchestrator**) quản lý luồng dữ liệu giữa các thư mục đầu ra của từng bước.

## 1. Cấu trúc thư mục dữ liệu

| Thư mục | Nội dung |
|---|---|
| `data/input/` | Ảnh flowchart gốc (PNG, JPG, JPEG, BMP) |
| `data/step1_preprocessed/` | Skeleton nhị phân 2 ngưỡng: `<stem>_preprocessed.png` (High-C) và `<stem>_preprocessed_low_c.png` (Low-C) |
| `data/step2_nodes/` | Thư mục làm việc của bước phát hiện node |
| `data/step3_edges/` | Thư mục làm việc của bước phát hiện cạnh |
| `data/step4_ocr/` | `<stem>_ocr.json` + `<stem>_ocr_debug.png` (khi bật OCR) |
| `data/output/` | `<stem>_graph.json` (đồ thị cuối) + `<stem>_result.png` (ảnh overlay) |

Các thư mục được tự tạo bởi `utils.io_helpers.ensure_dirs()` khi chạy `main.py`.

## 2. Chi tiết các bước

### Bước 1: Tiền xử lý (`steps/step1_preprocess.py`)

- **Mục tiêu:** chuyển ảnh sang grayscale, nhị phân hóa bằng **Adaptive Thresholding** (Gaussian, block size 15, `THRESH_BINARY_INV`), đóng hình thái học, rồi **skeletonize** (`skimage.morphology.skeletonize`) về nét 1-pixel.
- **Chiến lược Đa ngưỡng (Dual-C):**
  - **High-C** (`ADAPTIVE_THRESH_C_HIGH = 20`): lọc sạch nhiễu và bóng mờ → hỗ trợ Bước 2 tìm biên khối rõ nét.
  - **Low-C** (`ADAPTIVE_THRESH_C_LOW = 4`): giữ tối đa nét vẽ mảnh/mờ → hỗ trợ Bước 3 dò đường nối.
- **Đầu ra:** 2 ảnh skeleton lưu vào `data/step1_preprocessed/`.

### Bước 2: Nhận diện Node (`steps/step2_detect_nodes.py`)

Chia 2 pha, chạy trên skeleton High-C:

**Pha 2a — Phát hiện thô (`detect_nodes`):**
1. **Khử đầu mũi tên kín:** loại trước các contour tam giác lồi / chevron lõm (20–1500 px², 3–5 đỉnh) để không nhầm thành node.
2. **Contour Hierarchy (`RETR_TREE`):** node là các contour con trực thuộc contour bao ngoài lớn nhất (cộng thêm các contour top-level tách rời).
3. **Khớp hình học (`utils/ransac_shapes.py`):** thử 3 mô hình — hình chữ nhật (`minAreaRect` + kiểm tra góc ~90°), hình thoi trục-thẳng (dựng từ trung điểm bounding box), hình tròn (`minEnclosingCircle`) — chấm điểm bằng **tỷ lệ inlier** (khoảng cách điểm→biên < `RANSAC_INLIER_THRESH = 3 px`), chọn mô hình điểm cao nhất.
4. **Phân loại bổ trợ bằng Fill Ratio:** hình thoi lấp ~50% bounding box, hình chữ nhật ≥ 75%; kết hợp circularity (> 0.85 → Connector) và số đỉnh.
5. **Trích xuất lặp (Iterative Masking):** node tìm được lập tức bị tô đen với margin động tính từ **Distance Transform** (thích ứng độ dày nét), lặp tối đa `RANSAC_MAX_ITERATIONS = 20` vòng đến khi không còn ứng viên.
6. Lọc theo `MIN_NODE_AREA`, `MAX_NODE_WIDTH/HEIGHT × NODE_SIZE_TOLERANCE`, khử trùng lặp bằng IoU/containment. Node được sắp trên-xuống trái-phải và đánh ID `n1, n2, …`.

**Pha 2b — Tinh chỉnh (`finetune_nodes`):**
- **Lọc chồng lấp:** node giao đáng kể (> 10% diện tích nhỏ hơn) với ≥ 2 node khác — điển hình là khung "khổng lồ" bao trùm sơ đồ — bị gán `is_bad = True`, type `Bad`.

### Bước 3: Phát hiện đường nối (`steps/step3_detect_edges.py`)

Chia 2 pha, chạy trên skeleton Low-C:

**Pha 3a — Pairwise ROI Pathfinding (`detect_edges`):**
1. Tô đen toàn bộ vùng node hợp lệ (padding 3 px) để cô lập các nét nối; node `is_bad` bị loại khỏi xét duyệt.
2. Duyệt **từng cặp node** cách nhau ≤ 600 px; cắt ROI là bounding box hợp của 2 node (padding 5 px).
3. Một contour trong ROI tạo thành cạnh nếu **chạm cả 2 node** (có điểm rơi vào vùng snap 15 px quanh mỗi bounding box).
4. **Xác định hướng:** phát hiện trước các đầu mũi tên (contour 3–5 đỉnh, 30–800 px², ngoài node); mũi tên trong ROI gần node nào hơn → node đó là target.
5. Quỹ đạo contour được làm mịn (`approxPolyDP`, epsilon 2.0) và giữ trong trường `path` để trực quan hóa.

**Pha 3b — Hậu xử lý (`finetune_edges`):**
- **Nét vẽ mồ côi (Leftover):** xóa node hợp lệ + các contour cạnh đã dùng khỏi skeleton; contour còn sót có bounding box > 40 px được thêm vào danh sách node dưới dạng **Bad Node** để người dùng kiểm soát.

### Bước 4: Trích xuất văn bản (`steps/step4_ocr.py`)

- Chỉ chạy khi `ENABLE_OCR = True` (mặc định **False**).
- Cắt vùng node trên **ảnh gốc** với padding 8% (`OCR_ROI_PADDING_RATIO`), đưa qua **EasyOCR** (`["en", "vi"]`, GPU tùy chọn) ở chế độ paragraph.
- **Đầu ra:** cập nhật trường `text` cho từng node; lưu `<stem>_ocr.json` + ảnh debug vào `data/step4_ocr/`.

## 3. Bộ điều phối (`orchestrator.py`)

Luồng xử lý của `run_pipeline(image_path)`:

1. **Bước 1** → sinh 2 skeleton. Nếu Bước 2/3 không tìm thấy file skeleton tương ứng, ném `FileNotFoundError` ngay để kiểm soát chất lượng dữ liệu.
2. **Bước 2a → 2b → 3a → 3b → 4** chạy tuần tự.
3. **Hợp nhất:** gộp node (kèm `id`, `type`, `text`, `bbox`, `is_bad`) và edge (`source`, `target`, `label`) thành một dict đồ thị duy nhất; trường `path` của cạnh chỉ dùng nội bộ cho trực quan hóa, **không** ghi vào JSON cuối.
4. **Xuất kết quả:**
   - `data/output/<stem>_graph.json`
   - `data/output/<stem>_result.png` — vẽ bounding box node (xanh lá / đỏ cho Bad Node), tô magenta lên quỹ đạo cạnh thực tế bằng `fillPoly`, chấm đỏ tại node đích để chỉ hướng.

`run_all()` xử lý mọi ảnh trong `data/input/`; ảnh lỗi được log và bỏ qua.

**CLI (`main.py`):** `python main.py [IMAGE] [-v] [--max-node-size W,H]`.

## 4. Kiểm soát chất lượng (Human-in-the-loop)

Giao diện Streamlit (`streamlit run ui/app.py`) cho phép:
- Xem overlay kết quả, bật/tắt hiển thị Bad Node.
- Sửa type và text của từng node (Good/Bad tách nhóm riêng).
- Sửa source/target/label của cạnh, thêm cạnh mới.
- Lưu đè JSON và tự sinh lại ảnh trực quan hóa.

Mục tiêu: đạt độ chính xác 100% trước khi đưa dữ liệu vào hệ thống nghiệp vụ.

## 5. Hạn chế đã biết & hướng phát triển

- **Feedback loop đi vòng xa:** đường nối vòng ra ngoài bounding box hợp của 2 node bị crop khỏi ROI → mất cạnh. Hướng xử lý: Global Pathfinding (A*/BFS trên skeleton toàn cục) hoặc giãn nở ROI động.
- **Cặp node > 600 px** hiện bị bỏ qua (hằng số hard-code).
- **Nhãn cạnh (Yes/No)** chưa được OCR tự động — nhập tay qua UI.

Chi tiết phân tích và đánh giá thực nghiệm: xem [THESIS_NOTES.md](THESIS_NOTES.md).
