# VCMS

https://vcms-d5e48.web.app/

## Chức năng

- Kiểm duyệt nội dung video, hỗ trợ nhận diện các video có nội dung nhạy cảm liên quan tới chiến tranh, tai nạn, thiên tai, cờ bạc. Nhận diện các văn bản, từ ngữ xuất hiện trong video, từ đó người dùng có thể lọc ra những từ ngữ không phù hợp theo mục đích sử dụng của mình
- Hỗ trợ nhận diện văn bản, từ trong hình ảnh


## Tổng quan hệ thống

Sơ đồ tổng quan của hệ thống kiểm duyệt video như sau

![image.png](https://images.viblo.asia/ca93fce4-066e-4701-8edd-fe445b689a16.png)

Mô tả:

- Bước 1: Người dùng gửi request tới hệ thống kiểm duyệt, request có thể gồm link video youtube hoặc video định dạng .mp4.
- Bước 2: Hệ thống thực hiện đồng thời hai nhiệm vụ là Text recognition và Image classification. Với text recognition, cứ cách 5 giây ta lấy frame một lần để đưa ra dự đoán (do việc sử dụng toàn bộ frame là không mang nhiều ý nghĩa và làm chậm hệ thống). Frame sau khi đọc sẽ được đưa vào model DBNet và YoloV5 để đưa ra tọa độ bounding box của văn bản xuất hiện trong frame. Sau đó các ảnh chứa văn bản được cắt theo tọa độ bounding box sẽ được đưa vào model ABINet để đưa ra dự đoán là văn bản gì. Sau đó, các văn bản output được so sánh với bộ từ điển có sẵn để kiểm tra xem có xuất hiện từ không phù hợp hay không. Với image classification, cứ cách 1 giây ta lấy frame một lần để thực hiện dự đoán. Output của model này là ta có một list các class ứng với từng frame.
- Bước 3: Đóng gói output từ 2 module Text recognition và Image classification và phản hồi tới người dùng.

### Backend

Github repo:

Docker:

#### Các model AI sử dụng

##### Text recognition

Vì text xuất hiện trong video chủ yếu là scene text nên hệ thống sử dụng model DBNet và YoloV5 để cho tác vụ Text Detection. Với tác vụ Text Recognition, hệ thống sử dụng ABINet. Kết quả được đánh giá theo **WER ~ 2.4** trên các tập dữ liệu BKAI, VinText.

Tìm hiểu thêm về DBNet tại [link](/paper-reading/DBNet) và ABINet tại [link](/paper-reading/ABINet)

##### Image classification

Đây là phần phân loại hình ảnh trong video theo các class: Thiên tai, tai nạn, chiến tranh, cờ bạc và bình thường. Model sử dụng là Efficient B3.

Dưới đây là kết quả khi thực hiện train tới epoch 45, với bộ dataset được crawl theo google image.

![image.png](https://images.viblo.asia/210b6e74-e4c9-48e4-a2b8-c15a97cd464a.png)

#### APIs

APis được viết sử dụng framework Flask giúp hỗ trợ xây dựng API nhanh chóng.

### Frontend

Github repo: https://github.com/hieptran1812/VCMS

#### Tech stack sử dụng

Tech stack Frontend được sử dụng trong sản phẩm là ReactJS. Ngôn ngữ lập trình sử dụng là JavaScript. Ngoài ra các công cụ hỗ trợ để viết documentation là [Docusaurus](https://docusaurus.io/pt-BR/), [Stoplight]https://stoplight.io/). Đây là những công cụ mạnh trong việc xây dựng APIs Documentation, hỗ trợ xây dựng giao diện khoa học, bắt mắt đồng thời tối ưu được thời gian trong việc xây dựng Frontend. Frontend được hosting và auto deploy sử dụng Firebase.

#### Template tham khảo

Phần Frontend của sản phẩm được sử dụng template Frontend của trang https://dyte.io/. Đây là sản phẩm hỗ trợ xây dựng ứng dụng video stream, được đánh giá cao trên trang Product Hunt. Do giao diện được thiết kế bắt mắt, bố cục hợp lý và khoa học nên mình quyết định sử dụng template này cho sản phẩm VCMS.
