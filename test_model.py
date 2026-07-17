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

class ContextManaging:
    def __init__(self):
        self.current_context = None

    def _scan_keywords(self, user_text_lower, label_data):
        if "sub_intents" in label_data:
            for sub in label_data["sub_intents"]:
                for kw in sub["keywords"]:
                    if kw in user_text_lower:
                        return sub["response"]
        return None
    
    def set_new_contect_and_reply(self, user_text_lower,predicted_label, responses_data):
        self.current_context=predicted_label
        label_data=responses_data.get(predicted_label,{})   
        reply= self._scan_keywords(user_text_lower,label_data)
        return reply if reply else label_data.get("default", "Dạ em đã hiểu ý, nhưng chưa có kịch bản cho phần này.")
    
    def get_current_context(self):
        return self.current_context
    
    def process_message(self, user_text, confidence_score, predicted_label, responses_data):
        user_text_lower = user_text.lower()
        
        if self.current_context is None:
            if confidence_score>=-0.5:
                return self.set_new_contect_and_reply(user_text_lower,predicted_label,responses_data)
            else:
                return responses_data.get("fallback", {}).get("default", "Dạ em chưa hiểu ý anh/chị lắm.")
        else:
            if confidence_score>=0.0:
                return self.set_new_contect_and_reply(user_text_lower,predicted_label,responses_data)
            else:
                label_data = responses_data.get(self.current_context, {})
                reply = self._scan_keywords(user_text_lower, label_data)
                if reply:
                    return reply
                return "Dạ chi tiết này em chưa nắm rõ. Anh/chị vẫn đang quan tâm đến dịch vụ trước đó đúng không ạ?"
            

chat_session = ContextManaging()
print("=" * 60)
print(
    "🤖 CÔNG CỤ TEST MÔ HÌNH NHẬN DIỆN (NLU CORE), sử dụng dấu '|' nếu muốn hỏi nhiều câu hỏi cùng lúc"
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
    print("-" * 60)
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
        
        bot_reply=chat_session.process_message(question,max_score,predicted_label,responses)
        context_defy= chat_session.get_current_context()

        print(f"[{idx}] 👤 Khách: '{question}'")
        print(f"    🤖 Nhãn dự đoán : {predicted_label.upper()} ------ Nhãn context đang có : {context_defy.upper() if context_defy else None}")
        print(f"    📊 Độ tự tin    : {max_score:.2f}")
        print(f"    🤖 Bot phản hồi:\n{bot_reply}")
        if idx != len(questions):
            print("-" * 60)
    print("=" * 100)
