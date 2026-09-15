# RAG Project — ChromaDB + Python

Bộ khung cơ bản để test RAG với ChromaDB chạy dạng service (giống MySQL), tách riêng khỏi app.

## Cấu trúc

```
rag-project/
├── docker-compose.yml
├── README.md
└── app/
    ├── Dockerfile
    ├── requirements.txt
    ├── .env.example
    └── main.py
```

## Cách chạy

1. Tạo `app/.env` từ mẫu và điền OpenAI API key (bắt buộc cho bước embedding):

```bash
cp app/.env.example app/.env
# sửa OPENAI_API_KEY=... trong app/.env
```

2. Build và khởi động toàn bộ:

```bash
docker compose up --build
```

Lệnh này sẽ:
- Khởi động `chromadb` service, lắng nghe ở `localhost:8000`, dữ liệu lưu vào volume `chroma_data` (persistent, không mất khi restart container).
- Build và chạy `rag-app`, tự động chờ ChromaDB sẵn sàng (`healthcheck`), sau đó thêm dữ liệu mẫu và test truy vấn.

3. Xem log kết quả truy vấn ngay trên terminal.

4. Kiểm tra ChromaDB đang chạy độc lập:

```bash
curl http://localhost:8000/api/v1/heartbeat
```

## Dừng và dọn dẹp

```bash
docker compose down          # dừng container, giữ lại volume dữ liệu
docker compose down -v       # dừng và xóa luôn volume (mất dữ liệu)
```

## Bước tiếp theo

- Thay dữ liệu mẫu trong `main.py` bằng dữ liệu thật (PDF, docx...) — cần thêm bước chunk văn bản trước khi `collection.add()`.
- Thêm bước gọi LLM (ví dụ Claude API) sau khi lấy được context từ `collection.query()` để hoàn thiện pipeline generation.
- Embedding hiện dùng OpenAI `text-embedding-3-small` (gọi API, không cần torch); cần `OPENAI_API_KEY` trong `app/.env`.