import os
import time
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def wait_for_chroma(client, retries: int = 10, delay: int = 3):
    """Chờ ChromaDB service sẵn sàng trước khi thao tác."""
    for attempt in range(retries):
        try:
            client.heartbeat()
            print("ChromaDB đã sẵn sàng.")
            return
        except Exception:
            print(f"Đang chờ ChromaDB... ({attempt + 1}/{retries})")
            time.sleep(delay)
    raise RuntimeError("Không kết nối được tới ChromaDB service.")


def generate_answer(llm_client, question: str, contexts: list[str]) -> str:
    """Sinh câu trả lời tự nhiên dựa trên các document lấy được từ Chroma."""
    context_text = "\n".join(f"- {c}" for c in contexts)
    system_prompt = (
        "Bạn là trợ lý trả lời câu hỏi dựa trên NGỮ CẢNH được cung cấp. "
        "Chỉ dùng thông tin trong ngữ cảnh; nếu ngữ cảnh không đủ để trả lời, "
        "hãy nói rõ là không tìm thấy thông tin. Trả lời ngắn gọn, tự nhiên bằng tiếng Việt."
    )
    user_prompt = f"Ngữ cảnh:\n{context_text}\n\nCâu hỏi: {question}"
    resp = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def main():
    # Kết nối tới ChromaDB chạy dạng service (container riêng)
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    wait_for_chroma(client)

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Thiếu OPENAI_API_KEY. Copy app/.env.example thành app/.env và điền key."
        )

    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small",
    )

    collection = client.get_or_create_collection(
        name="rag_docs",
        embedding_function=embedding_fn,
    )

    # Dữ liệu mẫu để test nhanh
    sample_docs = [
        "ChromaDB là một vector database mã nguồn mở, dùng để lưu trữ embedding.",
        "RAG (Retrieval-Augmented Generation) kết hợp truy xuất dữ liệu và mô hình sinh văn bản.",
        "Docker Compose giúp quản lý nhiều container cùng lúc bằng một file cấu hình.",
    ]
    sample_ids = ["doc1", "doc2", "doc3"]

    existing = collection.get(ids=sample_ids)
    if not existing["ids"]:
        collection.add(documents=sample_docs, ids=sample_ids)
        print("Đã thêm dữ liệu mẫu vào collection.")
    else:
        print("Dữ liệu mẫu đã tồn tại, bỏ qua bước thêm.")

    # Test truy vấn
    query = "RAG là gì?"
    results = collection.query(query_texts=[query], n_results=2)

    print("\n--- Kết quả truy vấn ---")
    print(f"Câu hỏi: {query}")
    for doc, distance in zip(results["documents"][0], results["distances"][0]):
        print(f"- ({distance:.4f}) {doc}")

    # Bước generation: sinh câu trả lời như người, dựa trên context vừa truy xuất
    llm_client = OpenAI(api_key=OPENAI_API_KEY)
    answer = generate_answer(llm_client, query, results["documents"][0])
    print(f"\n--- Câu trả lời ({LLM_MODEL}) ---")
    print(answer)


if __name__ == "__main__":
    main()