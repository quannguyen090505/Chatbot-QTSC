import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("1.nạp dữ liệu huấn luyện từ file dataset_qtsc_cleaned.csv")
try:
    df = pd.read_csv(
        "./vncorenlp_model/dataset_qtsc_cleaned.csv",
        encoding="utf-8-sig",
        sep=":",
        engine="python",
    )
    df = df.dropna()
    print(f"-> Tổng số mẫu dữ liệu hợp lệ: {len(df)} câu hỏi.")
except Exception as e:
    print(f"Lỗi khi đọc file: {e}")
    exit()

X = df["Cleaned_Text"]
y = df["Label"]

print("2.vector hóa dữ liệu bằng TF-IDF N-grams")
vectorizer = TfidfVectorizer(
    max_df=0.85, analyzer="char_wb", ngram_range=(3, 4), min_df=2
)

X_tfidf = vectorizer.fit_transform(X)

print("3.chia tập dữ liệu: 80% để Học (Train), 20% để Thi thử (Test)")
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

print("4.huấn luyện thuật toán")
model = LinearSVC(C=0.67, dual="auto", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("=" * 50)
print("KẾT QUẢ HUẤN LUYỆN MÔ HÌNH")
accuracy = accuracy_score(y_test, y_pred)
print(f"=> ĐỘ CHÍNH XÁC TỔNG THỂ (ACCURACY): {accuracy * 100:.2f}%")

print("CHI TIẾT ĐIỂM SỐ TỪNG DỊCH VỤ:")
print(classification_report(y_test, y_pred, zero_division=0))

joblib.dump(model, "naive_bayes_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("-> Đã đóng gói 2 file naive_bayes_model.pkl và tfidf_vectorizer.pkl")
