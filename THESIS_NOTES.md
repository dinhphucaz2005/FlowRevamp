# Tái Cấu Trúc Đồ Thị Sơ Đồ Khối Bằng Thị Giác Máy Tính Cổ Điển

## 1. Đặt vấn đề

Trong bối cảnh chuyển đổi số, nhu cầu số hóa tài liệu giấy và sơ đồ khối (flowchart) thành dữ liệu có cấu trúc ngày càng quan trọng. Khi một sơ đồ được chuyển thành biểu diễn đồ thị (Graph/JSON), máy tính có thể tiếp tục chỉnh sửa, phân tích, sinh mã nguồn hoặc tích hợp với các hệ thống tự động khác.

Nhiều nghiên cứu hiện nay áp dụng Học sâu (Deep Learning) để nhận dạng sơ đồ khối với độ chính xác cao, nhưng đòi hỏi tập dữ liệu huấn luyện lớn, chi phí tính toán cao và phụ thuộc GPU.

Dự án **FlowRevamp** hướng tới một hệ thống số hóa sơ đồ khối hoạt động hoàn toàn trên CPU, sử dụng các thuật toán Thị giác máy tính cổ điển (Classical Computer Vision). Thay vì học đặc trưng từ dữ liệu, hệ thống khai thác trực tiếp các **đặc điểm hình học** của sơ đồ — đạt hiệu quả xử lý cao với chi phí tính toán rất thấp. Thành phần học máy duy nhất là EasyOCR cho bước nhận dạng chữ, và có thể tắt hoàn toàn.

---

## 2. Thiết kế hệ thống (Pipeline 4 bước)

Hệ thống được cấu trúc thành pipeline tuần tự gồm bốn giai đoạn, trong đó giai đoạn 2 và 3 chia thành pha phát hiện thô và pha tinh chỉnh:

**Bước 1 — Tiền xử lý (Preprocessing)**

Ảnh đầu vào được nhị phân hóa bằng **Adaptive Thresholding cục bộ** (Gaussian, block size 15) để khử bóng đổ và chiếu sáng không đều. Hệ thống áp dụng chiến lược **Đa ngưỡng song song (Dual-C)**:
- Phiên bản **High-C** (`C = 20`): lọc sạch nhiễu quanh hình học, cô lập khối rõ nét cho việc nhận dạng Node.
- Phiên bản **Low-C** (`C = 4`): bảo toàn các nét vẽ mảnh hoặc mờ phục vụ dò đường nối.

Cả hai phiên bản đều được đóng hình thái học rồi **làm mảnh nét về 1-pixel** (Morphological Skeletonization, cài đặt bằng `skimage.morphology.skeletonize`).

**Bước 2 — Phát hiện Node (Node Detection)**

Trên skeleton High-C, hệ thống dùng cấu trúc phân cấp đường biên (**Contour Hierarchy, `RETR_TREE`**) để trích các contour con bên trong khung bao ngoài — nhận xét then chốt là trong ảnh nhị phân flowchart, các khối và đường nối tạo thành một thành phần liên thông lớn, còn từng hình khối riêng lẻ xuất hiện dưới dạng contour con. Mỗi contour ứng viên được đưa qua bộ **khớp hình học kiểu RANSAC** (mục 3.1) để phân loại và định vị Node (Process, Decision, Terminal, Connector), kết hợp cơ chế **trích xuất lặp** (mục 3.3).

**Bước 3 — Phát hiện đường nối (Edge Detection)**

Thay vì Hough Transform toàn cục — dễ sai lệch khi nét vẽ cong hoặc gấp khúc — hệ thống áp dụng thuật toán **Pairwise ROI Pathfinding** trên skeleton Low-C: với mỗi cặp Node, cô lập một vùng ROI, tô đen (mask out) các vùng chứa Node, và kiểm tra sự tồn tại của một contour liên tục chạm cả hai Node. Hướng kết nối được xác định qua phát hiện đầu mũi tên (Arrowhead) trong vùng ROI.

**Bước 4 — Nhận dạng văn bản (OCR)**

Từng vùng Node hợp lệ được cắt trên **ảnh gốc** (thêm padding 8%) và đưa vào EasyOCR (tiếng Anh + tiếng Việt) để trích nội dung văn bản. Bước này là tùy chọn (`ENABLE_OCR`), giúp toàn bộ phần trích xuất cấu trúc chạy độc lập với mô hình học máy.

---

## 3. Các cải tiến thuật toán cốt lõi

### 3.1. Khớp hình học với chấm điểm inlier kiểu RANSAC

Với mỗi contour ứng viên, hệ thống dựng ba giả thuyết hình học tất định: hình chữ nhật (`cv2.minAreaRect`, kiểm tra bốn góc ≈ 90°), hình thoi trục-thẳng (dựng từ trung điểm bounding box), và hình tròn (`cv2.minEnclosingCircle`). Các giả thuyết được đánh giá theo tinh thần RANSAC: tính **tỷ lệ inlier** — phần trăm điểm contour cách biên mô hình dưới 3 px — và chọn mô hình có tỷ lệ cao nhất, loại bỏ nếu không mô hình nào vượt ngưỡng tối thiểu. Cách tiếp cận này giữ được tính bền vững với nhiễu của RANSAC nhưng tất định và nhanh hơn lấy mẫu ngẫu nhiên thuần túy, vì contour con sau tiền xử lý đã tương đối sạch.

### 3.2. Phân loại hình học bằng Fill Ratio

Thuật toán khớp hình đôi lúc khó phân biệt hình thoi (Decision) với hình chữ nhật (Process) bị xoay nhẹ. Hệ thống dùng chỉ số **Fill Ratio** = diện tích Contour / diện tích Bounding Box:
- Hình thoi có Fill Ratio xấp xỉ 50%.
- Hình chữ nhật có Fill Ratio từ khoảng 75% trở lên.

Kết hợp thêm số đỉnh xấp xỉ đa giác (`approxPolyDP`) và độ tròn (circularity > 0.85 → Connector), đặc trưng hình học thuần túy này phân loại Node với độ chính xác cao mà không cần mạng CNN.

### 3.3. Trích xuất lặp với margin động (Iterative Extraction)

Nếu chỉ quét contour một lượt, các Node lồng nhau hoặc dính sát đường nối sẽ bị bỏ sót. Hệ thống phát hiện theo nhiều vòng lặp: sau mỗi Node được xác nhận, contour của nó lập tức bị xóa (tô đen) khỏi ảnh nhị phân với **margin động tính từ Distance Transform** — margin tự thích ứng theo độ dày nét vẽ cục bộ, tránh xóa lẹm sang các nét lân cận. Vòng lặp tiếp theo quét phần ảnh còn lại, cho đến khi không còn đối tượng thỏa diện tích tối thiểu (tối đa 20 vòng).

### 3.4. Khử đầu mũi tên kín (Closed Arrowhead Removal)

Các đầu mũi tên đặc/kín là contour kín nhỏ, dễ bị nhận nhầm thành Node. Trước vòng lặp phát hiện, hệ thống quét và loại bỏ các contour nhỏ (20–1500 px²) có dạng tam giác lồi (solidity > 0.8) hoặc chevron lõm (4 đỉnh, solidity < 0.85).

### 3.5. Phát hiện "Node khổng lồ" và lọc chồng lấp

Một số sơ đồ chứa khung trang trí lớn bao phủ toàn ảnh; nếu coi đó là Node, nó sẽ che khuất mọi Node hợp lệ bên trong. FlowRevamp bổ sung bước đánh giá chồng lấp: một Node giao đáng kể với **từ hai Node hợp lệ trở lên** bị đánh dấu **Bad Node** (`is_bad = true`) và loại khỏi bước dò cạnh, nhưng vẫn giữ trong kết quả để người dùng kiểm tra.

### 3.6. Loại bỏ kết nối giả và phát hiện nét vẽ thừa (Leftover Paths)

Để triệt tiêu kết nối giả giữa các Node cùng hàng/cột, ROI trong bước dò cạnh được thu hẹp tối đa (padding chỉ 5 px) và điều kiện chạm Node được kiểm tra trên từng điểm contour (vùng snap 15 px). Sau khi hoàn tất dò cạnh, hệ thống tô đen toàn bộ Node và đường nối hợp lệ; nét vẽ mồ côi còn sót có kích thước đáng kể (> 40 px) được đánh dấu thành Bad Node để người dùng kiểm soát thủ công.

---

## 4. Mô hình Human-in-the-Loop (HITL)

Do sơ đồ viết tay đa dạng và nhiều nhiễu, đạt độ chính xác tuyệt đối 100% chỉ bằng thuật toán tự động là không khả thi. FlowRevamp được thiết kế theo mô hình **Human-in-the-Loop**:
- Hệ thống tự động thực hiện các tác vụ tính toán nặng (tiền xử lý, phát hiện node/edge, OCR).
- Người dùng hiệu chỉnh và kiểm tra cuối cùng qua giao diện Streamlit (`streamlit run ui/app.py`): sửa văn bản OCR, đổi loại Node, xử lý Bad Node, thêm/sửa liên kết bị thiếu hoặc sai hướng.

---

## 5. Hạn chế của Pairwise ROI & hướng phát triển

### 5.1. Ca thất bại: Bypass Loop / Feedback Loop

Trong flowchart phức tạp, đường phản hồi (feedback loop) nối từ bước cuối ngược lên bước đầu thường đi vòng ra sát biên ảnh để tránh đè lên chữ. Vì ROI được dựng bằng hộp giới hạn của hai Node nguồn–đích, phần đường nối đi vòng ra ngoài phạm vi này bị cắt bỏ (crop out), khiến thuật toán không tìm thấy kết nối liên tục.

### 5.2. Các hạn chế khác

- Cặp Node cách nhau quá 600 px hiện bị bỏ qua (hằng số cố định).
- Nhãn trên cạnh (Yes/No của Decision) chưa được OCR tự động.
- Hướng cạnh phụ thuộc vào việc nhận ra đầu mũi tên; mũi tên hở hoặc quá nhỏ khiến hướng mặc định về thứ tự cặp.

### 5.3. Giải pháp tương lai

- **Tìm đường toàn cục (Global Pathfinding):** chạy thuật toán loang (A*/BFS) trực tiếp trên skeleton nhị phân toàn cục từ Node A đến Node B.
- **Giãn nở ROI động (Dynamic ROI Expansion):** tự mở rộng vùng crop khi phát hiện nét vẽ đi sát rìa ROI hiện tại.
- **Giải pháp HITL:** cho phép người dùng nối trực tiếp trên giao diện Streamlit, tránh chi phí tính toán loang toàn cục.

---

## 6. Đánh giá thực nghiệm & so sánh với Baseline

Để định lượng đóng góp của các cải tiến, hệ thống tối ưu (`main.py`) được so sánh với pipeline Baseline thô sơ (`naive.py`) — giữ nguyên cấu trúc các bước nhưng loại bỏ mọi cải tiến:

| Tiêu chí | Baseline (`naive.py`) | FlowRevamp (`main.py`) |
| :--- | :--- | :--- |
| **Nhị phân hóa** | Otsu Threshold toàn cục | Adaptive Thresholding đa ngưỡng (High-C & Low-C) |
| **Dò tìm Node** | `findContours` một lượt + `approxPolyDP` | Iterative Masking + khớp hình học chấm điểm inlier |
| **Lọc hậu kỳ Node** | Không | Lọc chồng lấp (Bad Node) + khử arrowhead |
| **Dò tìm cạnh** | Hough toàn cục + nối thẳng tâm (Center Snapping) | Pairwise ROI Pathfinding |
| **Khả năng bám nét** | Thất bại khi nét cong / gấp khúc | Bám đúng mọi đường gấp khúc, zic-zắc |
| **Hậu xử lý cạnh** | Không | Gom nét vẽ mồ côi (Leftover) thành Bad Node |
| **Trực quan hóa** | Đường thẳng xuyên tâm (đường chéo ảo) | Tô quỹ đạo thực tế (`approxPolyDP` + `fillPoly`) |
| **Dung lượng model** | 0 MB | 0 MB (chưa tính EasyOCR tùy chọn) |
| **Thời gian xử lý** (đo nội bộ, chưa tính OCR) | ~10 ms — nhanh nhưng kết quả sai | ~15 ms — chính xác, vẫn thời gian thực trên CPU |

---

## 7. Đánh giá phản biện các ý tưởng tối ưu

Phần này tự đánh giá mức độ đóng góp và giới hạn của từng cải tiến, dựa trên kết quả thực nghiệm đối chứng với baseline (mục 6).

### 7.1. Các đóng góp cốt lõi

**Phả hệ contour `RETR_TREE` — node là contour con.** Đây là insight giá trị nhất của hệ thống. Thực nghiệm đối chứng cho thấy trực tiếp: baseline dùng `RETR_EXTERNAL` sụp đổ vì toàn bộ sơ đồ dính thành một thành phần liên thông (1 node, 0 cạnh); dùng `RETR_LIST` thì lẫn khối bao ngoài thành node khổng lồ. Nhận xét "khối và đường nối liên thông với nhau, nhưng ruột từng khối là contour con" biến bài toán tách node từ chỗ bất khả thi thành một phép tra cứu cấu trúc.

**Pairwise ROI Pathfinding.** Đảo ngược cách đặt bài toán: thay vì hỏi "các đoạn thẳng Hough thuộc cạnh nào" (bài toán gom nhóm, thất bại với nét cong/gấp khúc), hệ thống hỏi "giữa cặp node A–B có nét mực liên tục không" (bài toán kiểm tra tồn tại, dễ và bền vững). Trên ảnh thử nghiệm, phương pháp này tìm được 21 cạnh bám đúng quỹ đạo zic-zắc, so với 8 đường chéo sai của Hough toàn cục. Đánh đổi: độ phức tạp O(n²) theo số node — chấp nhận được với flowchart thực tế (< 50 node).

**Dual-C Thresholding.** Thừa nhận rằng không tồn tại một phép nhị phân hóa tối ưu cho mọi mục đích: node cần biên sạch nhiễu, line cần giữ nét mờ. Chi phí gần bằng 0 (chạy threshold hai lần), lợi ích rõ. Hạn chế: hai hằng số C = 20/4 hiện là lựa chọn kinh nghiệm; cần thí nghiệm độ nhạy tham số để biện luận đầy đủ.

**Fill Ratio.** Một đặc trưng hình học duy nhất (diện tích contour / bounding box ≈ 50% vs ≥ 75%) thay thế cả một bộ phân loại; khoảng cách giữa hai lớp đủ lớn để nhiễu không phá được với flowchart trục-thẳng.

### 7.2. Các lựa chọn thiết kế cần trình bày trung thực

**Khớp hình học "kiểu RANSAC".** Cài đặt thực tế **không lấy mẫu ngẫu nhiên**: mỗi contour được fit tất định (`minAreaRect`, `minEnclosingCircle`, rhombus từ trung điểm bounding box) rồi chấm điểm bằng tỷ lệ inlier. Đây là lựa chọn *có chủ đích* — contour con sau tiền xử lý đã tương đối sạch nên bước lấy mẫu ngẫu nhiên là thừa; phiên bản tất định nhanh hơn, ổn định và tái lập được. Thesis cần trình bày đúng như vậy ("đánh giá giả thuyết theo tiêu chí inlier của RANSAC, bỏ bước lấy mẫu ngẫu nhiên") thay vì gọi tắt là RANSAC.

**Iterative Masking.** Margin động theo Distance Transform là chi tiết tinh tế (margin cố định sẽ xóa lẹm nét mảnh hoặc xóa thiếu nét dày). Tuy nhiên thực nghiệm cho thấy vòng lặp hội tụ sau ~2 vòng; giới hạn 20 vòng chủ yếu là cơ chế an toàn, không phải yếu tố quyết định chất lượng.

**Bad Node — đánh dấu thay vì xóa.** Triết lý này khớp với mô hình HITL: hệ thống khoanh vùng nghi vấn (khung trang trí, vùng chữ tiêu đề, nét mồ côi) cho người dùng quyết định, thay vì tự tin loại bỏ sai. Hạn chế: quy tắc "chồng lấp ≥ 2 node" là heuristic cứng — một node hợp lệ lớn chứa các node lồng nhau (hiếm gặp) sẽ bị đánh dấu oan.

### 7.3. Giới hạn cần ghi nhận

- **Hướng cạnh phụ thuộc đầu mũi tên kín** (contour kín 30–800 px²): mũi tên hở (hai nét chữ V) không tạo contour kín nên cạnh mất hướng âm thầm — dự kiến là ca thất bại phổ biến với sơ đồ vẽ tay.
- **Các hằng số gắn với độ phân giải** (khoảng cách cặp 600 px, snap 15 px, leftover 40 px): ảnh scan độ phân giải khác sẽ lệch; hướng khắc phục là chuẩn hóa theo kích thước ảnh hoặc kích thước node trung vị.
- **Nhãn cạnh (Yes/No)** chưa được trích xuất tự động dù văn bản nằm ngay cạnh đường nối.
- **Feedback loop đi vòng xa** bị crop khỏi ROI (mục 5.1) — hạn chế cấu trúc của Pairwise ROI, không vá được bằng tham số; giải pháp global pathfinding (mục 5.3) đồng thời loại bỏ được cả giới hạn 600 px.

### 7.4. Nhận định chung

Giá trị của hệ thống không nằm ở từng kỹ thuật riêng lẻ mà ở **khung tư duy nhất quán**: mỗi cải tiến nhắm đúng một failure mode quan sát được của baseline và có thể chứng minh bằng ảnh đối chứng. Hai đóng góp cốt lõi là phân tích phả hệ contour và Pairwise ROI Pathfinding; Dual-C và Fill Ratio là cải tiến phụ trợ chắc chắn; phần khớp hình học cần được trình bày trung thực như một biến thể tất định của RANSAC để không trở thành điểm yếu khi phản biện.

---

## 8. Kết luận

FlowRevamp chứng minh rằng một hệ thống số hóa sơ đồ khối không nhất thiết phải phụ thuộc vào các mô hình Học sâu cồng kềnh để đạt hiệu quả thực tiễn. Việc kết hợp các thuật toán Thị giác máy tính cổ điển — nhị phân hóa thích nghi đa ngưỡng, phân tích phả hệ contour, khớp hình học chấm điểm inlier, và dò đường theo cặp ROI — cùng mô hình Human-in-the-Loop giúp hệ thống hoạt động ổn định, nhẹ trên CPU và có tính ứng dụng cao.
