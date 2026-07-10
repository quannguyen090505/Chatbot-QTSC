import pandas as pd
from collections import Counter
import re
from underthesea import word_tokenize

try:
    df = pd.read_csv("dataset_qtsc.csv", encoding="utf-8-sig", sep=":", engine="python")
except FileNotFoundError:
    print(f"Không tìm thấy file dataset_qtsc.csv")
    exit()

all_words = []
for text in df["Text"].dropna():
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = word_tokenize(text, format="text")
    all_words.extend(text.split())

WHITELIST_WORDS = []
try:
    with open("keywords.txt", "r", encoding="utf-8-sig") as keywords_file:
        for line in keywords_file:
            words = line.strip().lower()
            if words:
                WHITELIST_WORDS.append(words)
except FileNotFoundError:
    print(
        "Chưa có file keywords.txt, sau khi duyệt xong stopwords hãy kiểm tra lại và lọc thủ công các từ khóa nghiệp vụ"
    )

filtered_words = [
    word for word in all_words if word.replace("_", " ") not in WHITELIST_WORDS
]
filtered_counts = Counter(filtered_words)
top_stopwords = filtered_counts.most_common(300)
try:
    with open("stopwords.txt", "w", encoding="utf-8") as f:
        for word, _ in top_stopwords:
            f.write(f"{word}\n")

    print(f"Đã lưu danh sách stopwords vào lưu file stopwords.txt")
except Exception as e:
    print(f"Có lỗi xảy ra khi lưu file: {e}")
