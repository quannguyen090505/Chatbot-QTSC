import pandas as pd
import re
import os
import py_vncorenlp

try:
    df = pd.read_csv("dataset_qtsc.csv", encoding="utf-8-sig", sep=":")
    print(f"nạp {len(df)} câu hỏi huấn luyện.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file dataset_qtsc.csv, vui lòng kiểm tra lại thư mục.")
    exit()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rdrsegmenter = py_vncorenlp.VnCoreNLP(
    annotators=["wseg", "pos"], save_dir=os.path.join(BASE_DIR, "vncorenlp_model")
)
ALLOWED_POS_TAGS = ["N", "Nc", "Np", "Ny", "V"]

STOPWORDS = []
try:
    with open(
        os.path.join(BASE_DIR, "stopwords.txt"), "r", encoding="utf-8-sig"
    ) as stopwords_file:
        for line in stopwords_file:
            words = line.strip().lower()
            if words:
                STOPWORDS.append(words)
except FileNotFoundError:
    print(
        "Chưa có file stopwords.txt, thực hiện xử lý NLP lần đầu để làm dữ liệu đầu vào cho hàm find_stopwords.py "
    )


def clean_text_advanced(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    annotated_sentences = rdrsegmenter.annotate_text(text)

    final_tokens = []
    for sentence in annotated_sentences.values():
        for token_info in sentence:
            word = token_info["wordForm"]
            pos_tag = token_info["posTag"]

            if pos_tag in ALLOWED_POS_TAGS or word not in STOPWORDS:
                final_tokens.append(word)

    return " ".join(final_tokens)


df["Cleaned_Text"] = df["Text"].apply(clean_text_advanced)
df_final = df[["Label", "Cleaned_Text"]]
output_file = "dataset_qtsc_cleaned.csv"
df_final.to_csv(output_file, index=False, encoding="utf-8-sig", sep=":")
print(f"\nHoàn tất! Dữ liệu đã được làm sạch và lưu vào: {output_file}")
