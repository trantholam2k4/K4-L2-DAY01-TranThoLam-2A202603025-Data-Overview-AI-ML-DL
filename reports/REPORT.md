# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
6. Vì sao không được lấy `score` của model làm ground truth?

**Bài Làm**
1. Chép record hạng 1: `class_id`=468 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"

2. Record này nói về **cả bức ảnh**. Vì đây là bài toán Classification (Phân loại ảnh) dùng model yolo11n-cls.pt. Output của nó chỉ gán đúng 1 nhãn duy nhất đại diện cho nội dung toàn bộ bức ảnh, không hề có tọa độ hộp bao (Bounding Box: x, y, width, height) để chỉ đích danh vị trí của một vật thể nào cả.

3. Đi kèm theo Taxonomy / Checkpoint (yolo11n-cls.pt), mô hình hoàn toàn không tự nghĩ ra được. Vì Model được huấn luyện trên một tập dữ liệu chuẩn có sẵn (ví dụ ImageNet-1K gồm đúng 1000 lớp cố định). Model chỉ tính toán xác suất phân phối trên danh sách 1000 lớp cố định này, nó không có khả năng tự sáng tạo ra tên lớp mới.

4. Đáp án: Để đảm bảo tính toàn vẹn và tránh nhầm lẫn dữ liệu (Data Consistency & Lineage).

Giải thích thực tế:

    Cùng một chữ "car": trong bộ từ vựng (Taxonomy) của COCO nó có thể mang class_id = 2, nhưng sang bộ từ vựng của ImageNet hay Cityscapes nó lại mang class_id = 436 hoặc mang định nghĩa khác (xe taxi vs xe con).

    Nếu chỉ ghi mỗi tên lớp "car", khi chuyển giao dữ liệu giữa các đội ngũ hoặc nạp vào các model khác nhau sẽ bị xung đột, không biết dùng từ điển nào để giải mã. Ghi rõ cả 3 trường giúp hệ thống tự động ánh xạ (mapping) chính xác 100%.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, Guideline cần nói rõ điều gì để người gán nhãn chọn MỘT lớp?

    Đáp án: Guideline (Quy tắc gán nhãn) bắt buộc phải đưa ra Quy tắc ưu tiên (Priority Rule / Tie-breaking rule) rõ ràng.

    Cụ thể cần nói rõ:

        Quy tắc kích thước/trọng tâm: Chọn vật thể chiếm diện tích lớn nhất (Dominant object) ở trung tâm bức ảnh.

        Quy tắc thứ bậc ngữ cảnh: Ví dụ nếu có cả "đường phố" lẫn "chiếc xe taxi", quy tắc phải ghi rõ ưu tiên chọn bối cảnh chung (street) hay đối tượng phương tiện (cab/taxi).

        Nếu không có quy tắc này, mỗi annotator (người gán nhãn) sẽ chọn một kiểu theo cảm tính, làm dữ liệu bị nhiễu (inconsistent).

6. Vì sao KHÔNG ĐƯỢC lấy score của model làm Ground Truth?

    Đáp án: Vì Prediction của model bản chất chỉ là MỘT LỜI ĐOÁN MÒ CÓ XÁC SUẤT, không phải là Sự Thật Khách Quan (Ground Truth).

    Lý do sâu sắc:

        Model luôn có sai số (Bias/Error), nó có thể cực kỳ tự tin (Score = 0.99) nhưng thực tế đoán sai toét (ví dụ nhận nhầm con chó thành con sói).

        Ground Truth bắt buộc phải là nhãn do con người thẩm định và kiểm chứng dựa trên quy chuẩn (Guideline).

        Nếu lấy score của model làm ground truth, ta đang rơi vào cái bẫy Confirmation Bias (Học vẹt lỗi sai của chính model), model sau sẽ thừa hưởng và khuếch đại toàn bộ lỗi ngu ngốc của model trước mà không bao giờ tiến bộ được!


## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person" ; `score`=0.912624 ; `bbox_xyxy`=[385.33,69.24,498.92,348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Hộp bao (bounding box) bắt đầu từ góc trên-trái tại tọa độ (385.33, 69.24) và kết thúc tại góc dưới-phải ở tọa độ (498.92, 348.92). Hộp bao này có kích thước rộng 113.58 pixel và cao 279.68 pixel, bao trọn vẹn hình dáng của đối tượng người được nhận diện trong ảnh.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Ngưỡng 0.20: Có 17 vật thể (prediction).
Ngưỡng 0.60: Có 6 vật thể (prediction).

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ ngưỡng, mô hình sẽ trả về nhiều kết quả hơn (bao gồm cả các đối tượng có độ tự tin thấp). Điều này dẫn đến sự xuất hiện của nhiều Dương tính giả (False Positives) — các vùng không phải vật thể bị nhận diện nhầm. Do đó, Reviewer phải dành nhiều thời gian hơn để rà soát, lọc bỏ các hộp bao sai lệch hoặc gán nhãn lại nếu mô hình nhận diện không chính xác.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Quy tắc gán nhãn: Bounding box phải là hình chữ nhật nhỏ nhất ôm trọn vẹn đối tượng mục tiêu. Hộp bao phải sát mép vật thể, tránh bao gồm quá nhiều khoảng trống nền (background) xung quanh. Nếu vật thể bị cắt mép ảnh, hộp bao phải bao phủ phần diện tích hiển thị của vật thể đó cho đến mép ảnh.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

Vật thể: Người đàn ông đứng ở mép trái ngoài cùng của ảnh.
Tình trạng: Bị cắt mép (cropped) bởi khung hình phía bên trái.
Quyết định: Viết thành Quy tắc trong Guideline.
Vì sao? Vì đây là kịch bản rất phổ biến trong camera giám sát (vật thể di chuyển vào/ra khung hình). Thay vì đưa lên hỏi Lab Coach (Escalation) cho từng trường hợp, ta chỉ cần quy định rõ: "Nếu đối tượng bị cắt mép ảnh, vẽ hộp bao phủ lên phần diện tích nhìn thấy được của đối tượng đó" là xong. Đây là cách chuẩn hóa dữ liệu công nghiệp để đảm bảo tính đồng nhất (consistency).

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-004" ; `class_name`="potted plant" ; `score`=0.632235 ; số điểm polygon=`polygon_point_count`=100 ; 3–5 điểm đầu của `polygon_xy`=[1.0,2.0],[1.0,148.0],[3.0,148.0],[3.0,147.0],[4.0,146.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Bounding box chỉ cung cấp thông tin thô về tọa độ và kích thước tổng quát (hình chữ nhật), dẫn đến việc chứa nhiều khoảng trống nền (background noise). Polygon cung cấp thông tin độ phân giải pixel (pixel-level accuracy), giúp xác định biên giới chính xác của vật thể, loại bỏ các chi tiết không liên quan, từ đó giúp Robot thực hiện các thao tác gắp (grasping) hoặc tránh vật cản với độ chính xác cao hơn.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
Trả lời:

    Không phải Class ID: Vì Class ID dùng để phân loại đối tượng (ví dụ: tất cả các cái bát đều có cùng Class ID).

    Không phải Tracking ID: Vì đây là dữ liệu tĩnh (từng ảnh riêng lẻ), chưa có tính chất theo dõi xuyên suốt qua nhiều frame video.

    Bản chất: instance_id dùng để định danh riêng biệt cho từng đối tượng cụ thể trong một ảnh. Nếu ảnh có 3 cái bát giống hệt nhau, class_id đều là 'bowl', nhưng instance_id phải là 1, 2, 3 để hệ thống phân biệt được: đây là bát thứ nhất, đây là bát thứ hai. Nó là định danh cục bộ (local identifier) trong phạm vi bức ảnh đó.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Quy tắc: "Polygon phải bám sát đường viền vật thể. Tại những đoạn thẳng, dùng ít điểm neo. Tại những đoạn cong phức tạp, tăng mật độ điểm neo để đảm bảo đường viền mask không bị răng cưa. Đối với vật thể có lỗ hổng (ví dụ cái quai cầm của cái chảo), nếu yêu cầu gán nhãn chi tiết, phải vẽ lỗ hổng đó làm phần bị trừ (hole)."

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Quy tắc gán nhãn Polygon (Segmentation Guideline):

    Độ sát mép: Đường biên (mask) phải đi qua chính giữa các pixel ranh giới giữa vật thể và nền (background). Không được lấn sâu vào trong vật thể (gây thiếu dữ liệu) và không được bao phủ quá nhiều vùng nền (gây nhiễu dữ liệu).

    Độ chi tiết: Tại những đoạn đường cong (ví dụ: đường cong của cái bát, cái chảo), yêu cầu mật độ điểm neo (keypoints) cao hơn để đường biên trông mượt mà. Tại những đoạn đường thẳng (ví dụ: cạnh bàn), có thể giảm mật độ điểm neo để tối ưu hóa số lượng điểm dữ liệu.

    Xử lý che khuất: Nếu vật thể bị che khuất một phần bởi vật thể khác, chỉ gán nhãn phần đường biên có thể nhìn thấy được. Tại vị trí vật thể chồng lấn (overlap), đường biên phải được vẽ tách biệt rõ ràng dựa trên sự khác biệt về màu sắc/texture giữa hai vật thể.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Class Label (tên lớp) | Model đoán nhầm giữa các lớp tương đồng (ví dụ: cab/streetcar) | Chọn nhãn phù hợp nhất với guideline | Kiểm tra độ chính xác của nhãn (căn cứ vào class_id) |
| Phát hiện vật thể | Bounding Box [x,y,w,h] | Vật thể bị che khuất, bị cắt mép ảnh, vật nhỏ ở xa | Vẽ box khít sát vật thể, theo đúng guideline | Kiểm tra độ khít của box và số lượng vật thể bỏ sót |
| Instance segmentation | Polygon (đa giác) | Biên vật thể mờ, chồng lấn lên nhau | Vẽ đa giác bám sát mép vật thể | Kiểm tra độ mịn và khớp của mask với vật thể |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Quy tắc bảo vệ dữ liệu: "Không bao giờ để lộ thông tin cá nhân (họ tên, MSSV, email) trong file báo cáo hoặc dữ liệu đầu ra; chỉ sử dụng ảnh công khai (COCO dataset) cho bài tập và không upload dữ liệu nội bộ của doanh nghiệp lên môi trường public."

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Khi gặp dữ liệu lạ/ngoài phạm vi: "Dừng thao tác ngay lập tức, không tự ý gán nhãn hay bịa đặt nhãn. Ghi chú lại case khó và báo cáo cho Lab Coach/Mentor để được hướng dẫn xử lý theo đúng guideline của dự án."

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
