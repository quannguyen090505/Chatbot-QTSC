Đây là dự án hiện thực ứng dụng ChatBot hoạt động trên tập dữ liệu đóng của QTSC, Chatbot ứng dụng thuật toán các Machine Learning (LinearSVC/ComplementNB) và kỹ thuật trích xuất đặc trưng TF-IDF.
yêu cầu nền tảng:
- Python 3.8 - 3.11
- các thư viện cần thiết trong requirement.txt, sử dụng lệnh: pip install -r requirement.txt
Files dataset_qtsc.csv là dữ liệu đầu vào cần được làm sạch trước khi sử sụng để huấn luyện AI
Lần lượt chạy các phương thức:
- generating_stopwords.py, stopwords.py => sinh ra danh sách các stopwords có tần suất xuất hiện cao và cần được lược bỏ trong quá trình làm sạch dữ liệu
- keywords.txt => danh sách từ khóa nghiệp vụ cần dữ lại mà không bị trở thành stopwords
- preprocessing.py => phương thức thực hiện làm sạch dữ liệu dựa trên tập từ keywords và stopwords và sinh ra file dataset_qtsc.clean.csv đã được làm sạch
- train_model.py => phương thức huấn luyện AI dựa trên tập dữ liệu dataset_qtsc.clean.csv, sinh ra 2 file não bộ cốt lõi: tfidf_vectorizer.pkl (Từ điển TF-IDF đã huấn luyện)
naive_bayes_model.pkl (Bộ não AI  đã huấn luyện) cùng với kết quả của lần huấn luyện
- test_model.py => phương thức kiểm thử mô hình AI bằng cách đặt câu hỏi và nhận kết quả nhãn (Label) nhận biết được AI trả về (đối với họ thuật toán NB, kết quả độ tự tin là giá trị %, còn đối với họ thuật toán SVM, kết quả trả về là khoảng cách support vector, -0.5<= độ tự tin nghĩa là AI nhận dạng tích cực đối với các nhãn
- response.json => tập các câu trả lời ánh xạ cho các nhãn 
