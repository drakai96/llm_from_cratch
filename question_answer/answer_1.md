### 1. Phương pháp cơ bản để cập nhật trọng số  
- **Gradient Descent** là thuật toán nền tảng:  
  - **Công thức cập nhật trọng số**:  
    \[
    w = w - \eta \cdot \frac{\partial L}{\partial w}
    \]  
    - \( w \): Trọng số hiện tại.  
    - \( \eta \): Learning rate (tốc độ học).  
    - \( \frac{\partial L}{\partial w} \): Gradient của hàm mất mát \( L \) với trọng số \( w \).  

---

### 2. Các phương pháp cập nhật trọng số phổ biến  

#### **2.1. Batch Gradient Descent**  
- **Đặc điểm**:  
  - Sử dụng toàn bộ dữ liệu để tính toán gradient.  
  - Cập nhật trọng số sau khi tính gradient trên toàn bộ tập dữ liệu.  
- **Ưu điểm**:  
  - Gradient ổn định, hội tụ tốt.  
- **Nhược điểm**:  
  - Tốn tài nguyên khi dữ liệu lớn.  
  - Không phù hợp cho dữ liệu streaming.  

---

#### **2.2. Stochastic Gradient Descent (SGD)**  
- **Đặc điểm**:  
  - Cập nhật trọng số sau mỗi mẫu dữ liệu.  
- **Ưu điểm**:  
  - Cập nhật nhanh, phù hợp với dữ liệu lớn.  
- **Nhược điểm**:  
  - Gradient biến động lớn (không ổn định), hội tụ có thể không chính xác.  

---

#### **2.3. Mini-batch Gradient Descent**  
- **Đặc điểm**:  
  - Chia dữ liệu thành các batch nhỏ và cập nhật trọng số sau mỗi batch.  
- **Ưu điểm**:  
  - Cân bằng giữa tốc độ và ổn định.  
  - Thường được dùng phổ biến nhất.  
- **Kích thước batch (batch size)**:  
  - Giá trị thường từ 16 đến 512 (tùy thuộc vào GPU/CPU và dữ liệu).  

---

#### **2.4. SGD với Momentum**  
- **Công thức**:  
  \[
  v = \gamma v + \eta \frac{\partial L}{\partial w}, \quad w = w - v
  \]  
  - \( v \): Vận tốc (velocity), lưu thông tin gradient trước đó.  
  - \( \gamma \): Hệ số momentum (thường là 0.9).  
- **Ưu điểm**:  
  - Giảm dao động trong cập nhật trọng số, giúp hội tụ nhanh hơn.  

---

#### **2.5. Adaptive Gradient Descent Methods**  

##### **a. AdaGrad**  
- **Đặc điểm**:  
  - Điều chỉnh learning rate riêng cho từng trọng số, giảm tốc độ học theo thời gian.  
- **Công thức**:  
  \[
  w = w - \frac{\eta}{\sqrt{G_{t} + \epsilon}} \cdot \frac{\partial L}{\partial w}
  \]  
  - \( G_{t} \): Tổng bình phương gradient tích lũy.  
  - \( \epsilon \): Giá trị nhỏ để tránh chia cho 0.  

##### **b. RMSProp**  
- **Đặc điểm**:  
  - Giải quyết hạn chế của AdaGrad bằng cách lấy trung bình động (moving average) của gradient.  
- **Công thức**:  
  \[
  E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma) g_t^2, \quad w = w - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g
  \]  
  - \( \gamma \): Hệ số trung bình động (thường là 0.9).  

##### **c. Adam (Adaptive Moment Estimation)**  
- **Đặc điểm**:  
  - Kết hợp Momentum và RMSProp, rất phổ biến hiện nay.  
- **Công thức**:  
  \[
  m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
  \]  
  \[
  \hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
  \]  
  \[
  w = w - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
  \]  
  - \( \beta_1, \beta_2 \): Tham số momentum và trung bình động (thường là 0.9 và 0.999).  
- **Ưu điểm**:  
  - Hiệu quả cho các bài toán với gradient hiếm khi ổn định.  

---

### 3. Điều chỉnh học tốc (Learning Rate)  

#### **3.1. Learning Rate Decay**  
- Giảm dần learning rate theo thời gian hoặc epoch:  
  \[
  \eta_t = \frac{\eta_0}{1 + \lambda t}
  \]  
  - \( \lambda \): Hệ số giảm.  

#### **3.2. Cyclical Learning Rate**  
- Learning rate dao động giữa các giá trị nhỏ và lớn để thoát khỏi các vùng local minimum.  

---

### 4. Các lưu ý khi cập nhật trọng số  
- **Overfitting và Underfitting**:  
  - Quá trình cập nhật cần tránh mô hình overfitting.  
- **Gradient Vanishing/Exploding**:  
  - Sử dụng kỹ thuật chuẩn hóa (Batch Normalization) hoặc gradient clipping.  
- **Kiểm tra hội tụ**:  
  - Theo dõi giá trị hàm mất mát qua từng epoch.  