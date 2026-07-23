import joblib
import re
import warnings
import json
import time
from underthesea import word_tokenize
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore")

print("=" * 80)
print("🚀 KHỞI ĐỘNG HỆ THỐNG KIỂM THỬ NGỮ CẢNH SIÊU CẤP (MICRO-INTENT TESTING)")
print("=" * 80)

try:
    print("[1/3] Đang tải mô hình SVM và TF-IDF...")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("naive_bayes_model.pkl")
    
    print("[2/3] Đang tải CSDL Kịch bản (responses.json)...")
    with open("responses.json", "r", encoding="utf-8") as file:
        responses = json.load(file)
        
    print("[3/3] Đang tải Động cơ Ngữ nghĩa SBERT (Vui lòng chờ)...")
    sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("✅ Đã sẵn sàng!\n")
except FileNotFoundError as e:
    print(f"❌ [LỖI] Không tìm thấy file: {e}")
    exit()


def clean_input(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = word_tokenize(text, format="text")
    words = [w for w in text.split()]
    return " ".join(words)

class ContextManaging:
    def __init__(self):
        self.current_context = None
        self.vector_cache={}
        self.sbert_model = sbert_model

    def _get_keyword_vector(self, keyword):
            if keyword not in self.vector_cache:
                self.vector_cache[keyword] = self.sbert_model.encode(keyword)
            return self.vector_cache[keyword]
    
    def _scan_semantic_intent(self, user_text, label_data):
        if "sub_intents" not in label_data:
            return (None,0.0)
        user_vector = self.sbert_model.encode(user_text)
        best_score = 0
        best_response = None

        for sub in label_data["sub_intents"]:
            for keyword in sub["keywords"]:
                kw_vector=self._get_keyword_vector(keyword)
                score = util.cos_sim(user_vector, kw_vector).item()
                if score > best_score:
                    best_score = score
                    best_response = sub["response"]
        if best_score >= 0.50:
            return (best_response, best_score)
        return (None,best_score)
    
    def set_new_context_and_reply(self, user_text_lower, predicted_label, responses_data):
        self.current_context = predicted_label
        label_data = responses_data.get(predicted_label, {})   
        reply, score = self._scan_semantic_intent(user_text_lower, label_data)
        return (reply if reply else label_data.get("default", "Fallback Default"),score)
    
    def get_current_context(self):
        return self.current_context
    
    def process_message(self, user_text, confidence_score, predicted_label, responses_data):
        user_text_lower = user_text.lower()
        
        if self.current_context is None:
            if confidence_score >= -0.6:
                return self.set_new_context_and_reply(user_text_lower, predicted_label, responses_data)
            else:
                return (responses_data.get("fallback", {}).get("default", "Fallback Global"),0.0)
        else:
            if confidence_score >= 0.2:
                return self.set_new_context_and_reply(user_text_lower, predicted_label, responses_data)
            else:
                label_data = responses_data.get(self.current_context, {})
                reply,score = self._scan_semantic_intent(user_text_lower, label_data)
                if reply:
                    return (reply,score)
                return ("Context Fallback", score )

def get_description_from_response(reply, responses_data):
    if reply == "Fallback Global" or "chưa hiểu rõ ý" in reply.lower():
        return "Fallback Global"
    if reply == "Context Fallback" or "vẫn đang quan tâm" in reply.lower():
        return "Context Fallback"
    if reply == "Fallback Default":
        return "Default"

    for intent, data in responses_data.items():
        if reply == data.get("default", ""):
            return "Default"
        for sub in data.get("sub_intents", []):
            if reply == sub.get("response", ""):
                return sub.get("description", "No Description")
    return "Unknown"

TEST_FILE = "context_testcases.json"
try:
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        TEST_SESSIONS = json.load(f)
except Exception as e:
    print(f"❌ [LỖI] Không thể đọc file kịch bản {TEST_FILE}: {e}")
    exit()

total_turns = 0
passed_turns = 0

print("BẮT ĐẦU CHẠY CÁC PHIÊN HỘI THOẠI...\n")
time.sleep(1)

for session_idx, session_turns in enumerate(TEST_SESSIONS, 1):
    print(f"🔹 SESSION {session_idx:02d}")
    bot = ContextManaging()
    
    for turn_idx, turn in enumerate(session_turns, 1):
        total_turns += 1
        user_text = turn["user"]
        expected_context = turn["expected_context"]
        expected_desc = turn["expected_description"]
        
        cleaned_text = clean_input(user_text)
        text_vector = vectorizer.transform([cleaned_text])
        predicted_label = model.predict(text_vector)[0]
        
        try:
            decision_scores = model.decision_function(text_vector)[0]
            confidence = decision_scores if isinstance(decision_scores, float) else max(decision_scores)
        except Exception:
            confidence = 1.0
            
        reply,score = bot.process_message(user_text, confidence, predicted_label, responses)
        actual_context = bot.get_current_context()
        
        actual_desc = get_description_from_response(reply, responses)
        
        context_passed = (actual_context == expected_context)
        desc_passed = (actual_desc == expected_desc)
        
        turn_passed = context_passed and desc_passed
        if turn_passed:
            passed_turns += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            
        print(f"   [{turn_idx}] Khách: '{user_text}'")
        print(f"       -> Ngữ cảnh  : {actual_context}, độ tự tin: {confidence:.2f}, (Kỳ vọng: {expected_context})")
        print(f"       -> Trả lời   : {actual_desc}, độ nhân dạng: {score:.2f} (Kỳ vọng: {expected_desc})")
        print(f"       -> Kết quả   : {status}\n")

print("=" * 80)
print(f"🎯 KẾT QUẢ KIỂM THỬ ĐỊNH TUYẾN VI MÔ (MICRO-INTENT ROUTING REPORT)")
accuracy = (passed_turns / total_turns) * 100
print(f"TỶ LỆ CHÍNH XÁC: {accuracy:.2f}% ({passed_turns}/{total_turns} Lượt chat lấy đúng kịch bản)")
print("=" * 80)