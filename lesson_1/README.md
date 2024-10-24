# Hướng dẫn chạy code

## Step 1: Khởi tạo môi trường

Trước tiên, cần tạo môi trường ảo để quản lý các thư viện và tránh xung đột phiên bản. Thực hiện lệnh sau để tạo môi trường mới với Python 3.11:

```bash
conda create -n aicoaching python=3.11
```
Sau khi tạo xong, kích hoạt môi trường bằng lệnh:
```bash
conda activate aicoaching
```
##Step 2: Cài đặt các thư viện

Để cài đặt tất cả các thư viện cần thiết cho dự án, sử dụng lệnh sau:
```bash
pip install -r requirements.txt
```
Step 3: Chạy ứng dụng

Sau khi cài đặt các thư viện, chạy ứng dụng bằng cách sử dụng shell script sau:
```bash
sh run.sh
```
