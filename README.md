# Chess_AI

## Demo video link: 
### https://drive.google.com/file/d/1PW6fcMj1puivHYwfLKtQ5t1gzGXggikH/view?usp=sharing

## Giới thiệu
Chess_AI là một dự án trí tuệ nhân tạo chơi cờ vua tiên tiến, kết hợp nhiều phương pháp AI khác nhau từ cổ điển đến hiện đại. Hệ thống được thiết kế với mục tiêu tạo ra một AI có khả năng chơi cờ ở nhiều cấp độ khác nhau, từ người mới bắt đầu đến cao thủ, đồng thời cung cấp giao diện thân thiện với người dùng.

### Đặc điểm nổi bật
- Kết hợp nhiều phương pháp AI hiện đại
- Khả năng điều chỉnh độ mạnh linh hoạt
- Giao diện đa nền tảng (Desktop & Web)
- Tương thích với các phần mềm cờ vua chuyên nghiệp qua giao thức UCI
- Tích hợp công cụ phân tích và học tập

### Tính năng chính

#### 1. Phương pháp AI
##### Basic Evaluation
- Đánh giá vật chất (quân cờ)
  - Giá trị chuẩn cho từng loại quân
  - Điều chỉnh theo giai đoạn ván đấu
- Bảng giá trị vị trí (Piece-Square Tables)
  - Tối ưu hóa vị trí từng quân
  - Thay đổi theo giai đoạn (khai cuộc, trung cuộc, tàn cuộc)
- Đánh giá cấu trúc tốt
  - Tốt thông
  - Tốt đôi
  - Tốt cô lập
  - Chuỗi tốt
- Tính điểm di chuyển của quân
  - Kiểm soát trung tâm
  - Tính linh động của quân

##### Advanced Evaluation
- Phân tích an toàn vua
  - Đánh giá lá chắn tốt
  - Phân tích các đường tấn công
  - Nhận diện các mẫu tấn công nguy hiểm
- Đánh giá cấu trúc phức tạp
  - Phân tích cột mở và nửa mở
  - Đánh giá kiểm soát đường chéo
  - Nhận diện các điểm yếu trong trận địa
- Đánh giá động
  - Tính toán các đe dọa
  - Đánh giá áp lực tấn công
  - Phân tích khả năng phòng thủ

#### 2. Giao diện người dùng
##### Desktop Interface (Pygame)
- Giao diện đồ họa hiện đại
- Hỗ trợ kéo thả quân cờ
- Hiển thị nước đi gợi ý
- Phân tích thế cờ trực quan
- Tùy chọn góc nhìn (2D/3D)

##### Web Interface (NiceGUI)
- Truy cập từ mọi thiết bị
- Responsive design
- Tích hợp công cụ phân tích
- Xem lại ván đấu
- Chia sẻ thế cờ dễ dàng

#### 3. Tính năng nâng cao
##### Opening Book
- Database khai cuộc phong phú
- Tự động học và cập nhật
- Thống kê tỉ lệ thắng
- Đa dạng hệ thống khai cuộc

##### Endgame Tablebases
- Hỗ trợ Syzygy tablebases
- Chơi tàn cuộc hoàn hảo
- Tối ưu hóa bộ nhớ
- Tự động nén/giải nén

##### Phân tích và học tập
- So sánh với Stockfish
- Phân tích sai lầm
- Gợi ý cải thiện
- Thống kê hiệu suất

### Yêu cầu hệ thống
#### Phần cứng tối thiểu
- CPU: 2 cores trở lên
- RAM: 4GB
- Dung lượng ổ cứng: 2GB cho cài đặt cơ bản
- GPU: Không bắt buộc, khuyến nghị cho neural network

#### Phần mềm
- Hệ điều hành: Windows 10+, Linux, macOS
- Python 3.8+ với các thư viện:
  - chess~=1.11.2: Xử lý luật cờ vua
  - pygame-ce: Giao diện đồ họa
  - numpy~=2.1.3: Tính toán ma trận
  - torch~=2.7.0: Deep learning
  - tensorflow~=2.19.0: Deep learning
  - nicegui~=2.16.1: Web interface
  - Các thư viện phụ thuộc khác trong requirements.txt

### Cài đặt
1. Clone repository:
```bash
git clone https://github.com/your-username/Chess_AI.git
cd Chess_AI
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

3. (Tùy chọn) Tải Syzygy tablebases:
```bash
# Tạo thư mục syzygy
mkdir syzygy
cd syzygy
# Tải và giải nén tablebases
wget https://tablebase.lichess.ovh/tables/standard/3-4-5/3-4-5.zip
unzip 3-4-5.zip
```

### Hướng dẫn sử dụng

#### 1. Giao diện đồ họa
```bash
python -m algo.ui
```
- Sử dụng chuột để di chuyển quân
- Phím tắt:
  - R: Đảo ngược bàn cờ
  - U: Đi lại
  - N: Ván mới
  - A: Bật/tắt phân tích
  - S: Lưu ván đấu

#### 2. Web interface
```bash
python -m algo.ui --web
```
- Truy cập http://localhost:8080
- Hỗ trợ nhiều người chơi cùng lúc
- Tích hợp chat và chia sẻ

#### 3. UCI engine
```bash
python -m algo.main
```
- Tương thích với các GUI cờ vua: Arena, SCID, Fritz
- Hỗ trợ các lệnh UCI tiêu chuẩn
- Tùy chỉnh thông số engine

### Cấu hình nâng cao
- Điều chỉnh sức mạnh trong `config.py`
- Tùy chỉnh tham số tìm kiếm
- Cấu hình neural network
- Quản lý bộ nhớ cache

## Đóng góp

### Vũ Huy Công: Evaluation, Opening-book,endgame-table, Search, Evaluation
### Nguyễn Đức Dũng : UI, Search
### Nguyẽn Đức Duy : Bitboard, magic-bitboard,Mover-ordering, evaluation
### Nguyễn Đức Hưng : Zobrist Hashing, Transposition Table, Performance testing
 
