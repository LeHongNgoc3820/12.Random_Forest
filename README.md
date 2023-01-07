# Random Forest
### Nội dung
1. Giới thiệu
2. Các ứng dụng
3. Thuật toán
4. Ưu/khuyết điểm
5. Xây dựng Random Forest

## 1. Giới thiệu
+ Nhược điểm của bài toán Decision Tree là chúng có khuynh hướng overfit dữ liệu huấn luyện. Tuy nhiên, **Random Forest là một cách để giải quyết vấn đề này.**
+ Một Random Forest về bản chất là một tập hợp các cây quyết định, trong đó, mỗi cây hơi khác so với các cây khác.
+ Ý tưởng đằng sau Randomn Forest là mỗi cây có thể làm một công việc dự đoán tương đối tốt, nhưng có khả năng sẽ overfit một phần dữ liệu. Nếu ta xây dựng nhiều cây, tất cả các cây đều hoạt động tốt và vượt trội theo nhiều cách khác nhau, ta có thể giảm số lượng overfit bằng cách lấy trung bình kết quả của chúng.
+ Điều này giúp giảm overfitting, trong khi vẫn giữ lại sức mạnh dự đoán của cây.
+ Để thực hiện chiến lược này, ta cần xây dựng nhiều cây quyết định. Mỗi cây làm một công việc chấp nhận được để dự đoán target và nên khác với các cây khác.
+ Ten gọi Random Forest bắt nguồn từ việc đưa ngẫu nhiên mẫu vào cây để đảm bảo mỗi cây khác nhau.
+ Có hai cách để các cây trong random forest được chọn ngẫu nhiên:
    + Bằng cách chọn các điểm dữ liệu được sử dụng để xây dựng một cây.
    + Bằng cách chọn các tính năng trong mỗi split test.

### So sánh với Decision Tree
+ Random Forest là một tập hợp của nhiều Decision Tree
+ Decision Tree có thể bị ảnh hưởng bởi overfitting, còn Random Forest ngăn overfitting bằng cách tạo cây trên các tập con ngẫu nhiên.
+ Decision Tree tính toán nhanh hơn.
+ Random Forest khó giải thích, trong khi Decision Trê có thể diễn giải dễ dàng và có thể chuyển đổi thành quy tắc.

## 2. Các ứng dụng
+ Phân loại các phương tiện giao thông được phát hiện
+ Phân loại ảnh viễn thám (remote sensing)
+ Dự báo xu hướng cổ phiếu trên thị trường chứng khoán (Stock Trend Forecasts)
+ Video classification
+ Image classification
+ ...

## 3. Thuật toán
Thuật toán này làm việc theo 4 bước:
+ Chọn các mẫu ngẫu nhiên từ tập dữ liệu cung cấp.
+ Xây dựng cây quyết định cho các mẫu được chọn và nhận kết quả dự đoán từ mỗi cây quyết định
+ Thực hiện bỏ phiếu cho từng kết quả dự đoán.
+ Chọn kết quả dự đoán có nhiều phiếu nhất là dự đoán cuối cùng.

## 4. Ưu/khuyết điểm
### Ưu điểm
+ Random Forest được coi là một phương pháp chính xác và mạnh mẽ vì số lượng các decision tree tham gia vào quá trình này.
+ Random Forest không bị vấn đề overtfitting vì nó lấy trung bình của tất cả các dự đoán, trong đó huỷ bỏ những bias.
+ Thuật toán có thể được sử dụng trong cả **classification** và **regression**.
+ Random Forest cũng có thể xử lý các giá trị còn thiếu. Có hai cách để xử lý các giá trị thiếu này:
    + Sử dụng các giá trị median để thay thế các biến liên tục
    + Tính toán mức trung bình gần kề, proximity-weighted average của các giá trị bị thiếu
+ Ta có thể nhận thấy tầm quan trọng của tính năng tương đối, giúp chọn các tính năng đóng góp nhiều nhất cho quá trình phân loại.

>### Làm thế nào để tìm thuộc tính quan trọng?
>+ Random Forest cung cấp một chỉ số lựa chọn tính năng tốt. Scikit-learn cung cấp thêm một biến với model, cho thấy tầm quan trọng hoặc đóng góp tương đối của từng tính năng trong dự đoán.
>+ Nó tự động tính toán điểm liên quan của từng tính năng trong quá trình đào tạo. Sau đó, nó cân đối mức độ liên quan xuống sao cho tổng của tất cả các điểm là 1. Điểm số này sẽ giúp ta chọn các tính năng quan trọng nhất và bỏ đi các tính năng ít quan trọng trong việc xây dựng model.

+ Random Forest sử dụng tầm quan trọng của gini hoặc giảm trung bình trong tạp chất - Mean Decrease in Impurity (MDI) để tính toán tầm quan trọng của từng tính năng.
+ Tầm quan trọng Gini còn được gọi là tổng giảm trong tạp chất của nút (node impurity). Đây là cách model phù hợp hoặc giảm độ chính xác của mô hình khi bạn bỏ đi một biến. Độ lớn càng lớn thì biến số càng có ý nghĩa. Ở đây, tổng giảm trung bình là một tham số quan trọng cho việc lựa chọn biến. Chỉ số Gini có thể mô tả sức mạnh giải thích tổng thể của các biến.
### Khuyết điểm
+ Random Forest thực hiện chậm trong việc dự đoán bởi vì nó có nhiều decision tree. Bất cứ khi nào nó đưa ra dự đoán, tất cả các cây trong rừng phải đưa ra dự đoán cho cùng một đầu vào cho trước và sau đó thực hiện bỏ phiếu trên đó. Toàn bộ quá trình này tốn thời gian.
+ Mô hình khó hiểu hơn so với decision tree vì với decision tree, ta có thể dễ dàng đưa ra quyết định bằng cách đi theo đường dẫn trong cây.

## 5. Xây dựng Random Forest
Dùng `sklearn.ensemble.RandomForestClassifier` hoặc `sklearn.ensemble.RandomForestRegressor`

### Các bước thực hiện
+ Chọn model sẽ sử dụng là **RandomForestClassifier** hay **RandomForestRegressor**
+ Tạo một tập dữ liệu feature và một tập target chứa các nhãn/giá trị cho các thực thể
+ Chia dữ liệu thành train-test
+ Áp dụng mô hình
+ Hoàn chỉnh model cho training data
+ Sử dụng model hoàn chỉnh (fitted model) cho dữ liệu chưa biết (unseen data)
+ Đánh giá độ chính xác
