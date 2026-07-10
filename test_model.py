import joblib
import re
from underthesea import word_tokenize
import warnings
import json

warnings.filterwarnings("ignore")  # Tắt các cảnh báo lặt vặt

try:
    print("Đang tải 'Bộ não AI' (Mô hình & Từ điển)...")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("naive_bayes_model.pkl")
    print("[THÀNH CÔNG] Bot đã sẵn sàng nhận câu hỏi!\n")
except FileNotFoundError:
    print(
        "[LỖI] Không tìm thấy file .pkl. Hãy chắc chắn bạn đã chạy train_model.py trước."
    )
    exit()

try:
    print("3. Đang nạp kịch bản phản hồi từ responses.json...")
    with open("responses.json", "r", encoding="utf-8") as file:
        responses = json.load(file)
    print("-> [THÀNH CÔNG] Hệ thống đã sẵn sàng!\n")
except FileNotFoundError:
    print("[LỖI] Không tìm thấy file responses.json!")
    exit()

stop_words = [
    "là",
    "của",
    "và",
    "có",
    "không",
    "thì",
    "mà",
    "để",
    "với",
    "cho",
    "những",
    "các",
    "một",
    "như",
    "này",
    "được",
    "vậy",
    "cái",
    "đi",
    "tôi",
    "mình",
    "bạn",
    "anh",
    "chị",
    "em",
    "dạ",
    "ạ",
    "vâng",
    "xin",
    "chào",
    "nhé",
    "nha",
    "ơi",
    "hả",
    "trời",
    "cho_hỏi",
]


def clean_input(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = word_tokenize(text, format="text")
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)


print("=" * 60)
print(
    "🤖 CÔNG CỤ TEST NHANH MÔ HÌNH NHẬN DIỆN (NLU CORE), sử dụng dấu '|' nếu muốn hỏi nhiều câu hỏi cùng lúc"
)
print("Gõ 'quit' hoặc 'exit' để thoát.")
print("=" * 60)

while True:
    user_input = input("\n👤 Bạn (Khách hàng): ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Đã thoát chương trình test.")
        break

    if not user_input.strip():
        continue

    questions = [q.strip() for q in user_input.split("|") if q.strip()]
    print("-" * 40)
    for idx, question in enumerate(questions, 1):
        cleaned_text = clean_input(question)
        text_vector = vectorizer.transform([cleaned_text])
        predicted_label = model.predict(text_vector)[0]

        decision_scores = model.decision_function(text_vector)[0]
        try:
            decision_scores = model.decision_function(text_vector)[0]
            max_score = (
                decision_scores
                if isinstance(decision_scores, float)
                else max(decision_scores)
            )
        except Exception:
            max_score = 1.0

        if max_score < -0.5:
            bot_reply = responses.get("fallback")
        else:
            bot_reply = responses.get(
                predicted_label,
                f"[HỆ THỐNG]: Đã nhận diện được nhãn '{predicted_label}' nhưng chưa viết kịch bản trong file JSON.",
            )
        print(f"[{idx}] 👤 Khách: '{question}'")
        print(f"    🤖 Nhãn dự đoán : {predicted_label.upper()}")
        print(f"    📊 Độ tự tin    : {max_score:.2f}")
        print(f"    🤖 Bot phản hồi: {bot_reply}")
        if idx != len(questions):
            print("-" * 60)
    print("=" * 100)
