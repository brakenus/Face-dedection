import cv2
import mediapipe as mp
import numpy as np
import os


def find_file_in_directory(directory, filename):
    """Belirtilen klasörde dosyayı arar ve tam yolunu döndürür."""
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None

def initialize_gender_age_model(base_dir="C:/Users/BRAKENUS/Desktop/PY"):
    """Cinsiyet ve yaş tahmini modellerini başlatır."""
    # Dosya adlarını belirtiyoruz
    age_prototxt = "deploy_age.prototxt"
    age_caffemodel = "age_net.caffemodel"
    
    # PY klasöründe dosyaları arıyoruz
    age_prototxt_path = find_file_in_directory(base_dir, age_prototxt)
    age_caffemodel_path = find_file_in_directory(base_dir, age_caffemodel)
    
    # Dosyaların bulunup bulunmadığını kontrol ediyoruz
    if not age_prototxt_path or not age_caffemodel_path:
        raise FileNotFoundError(f"{age_prototxt} veya {age_caffemodel} bulunamadı!")
    
    # Modelleri yükleme
    age_net = cv2.dnn.readNetFromCaffe(age_prototxt_path, age_caffemodel_path)
    return age_net

# Kullanım
try:
    age_net = initialize_gender_age_model()
    print("Age net başarıyla yüklendi.")
except Exception as e:
    print(f"Hata: {e}")





def get_videos_from_folder(folder_path):
    # "yüz" kelimesini içeren mp4 dosyalarını bul
    videos = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.mp4') and "yüz" in file]
    return videos

def initialize_mediapipe():
    # Mediapipe ile yüz tespiti modelini başlat
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    return face_detection

def initialize_gender_age_model():
    # Cinsiyet ve yaş tespiti için DNN modelini başlat
    gender_net = cv2.dnn.readNetFromCaffe(
        "deploy_gender.prototxt", "gender_net.caffemodel")
    age_net = cv2.dnn.readNetFromCaffe(
        "deploy_age.prototxt", "age_net.caffemodel")
    return gender_net, age_net

def get_age_and_gender(frame, gender_net, age_net):
    # Cinsiyet ve yaş tahmini için DNN modelini kullan
    blob = cv2.dnn.blobFromImage(frame, 1, (224, 224), (104, 117, 123), swapRB=False, crop=False)
    
    # Cinsiyet tahmini
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = "Male" if gender_preds[0].argmax() == 0 else "Female"
    
    # Yaş tahmini
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = int(age_preds[0].argmax())
    
    return age, gender

def draw_text(frame, text, position, font_scale=1, color=(255, 255, 255)):
    # Videonun üzerine metin ekle
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    cv2.putText(frame, text, position, font, font_scale, color, thickness)

def process_video(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Giriş video dosyası bulunamadı: {input_path}")
        return
    
    # Video yakalama
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Video açılamadı: {input_path}")
        return
    
    # Video özelliklerini al
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Çıkış formatı (mp4)
    
    # Video yazıcıyı başlat
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Mediapipe'yi başlat
    face_detection = initialize_mediapipe()
    gender_net, age_net = initialize_gender_age_model()
    
    print("İşlem başlıyor...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Renkleri BGR'den RGB'ye çevir
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Yüz tespiti
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            # Her bir tespit edilen yüz için cinsiyet ve yaş tahmini yap
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x_min = int(bboxC.xmin * frame_width)
                y_min = int(bboxC.ymin * frame_height)
                box_width = int(bboxC.width * frame_width)
                box_height = int(bboxC.height * frame_height)

                # Yüz bölgesini kırp
                face_roi = frame[y_min:y_min + box_height, x_min:x_min + box_width]

                # Cinsiyet ve yaş tahmini yap
                age, gender = get_age_and_gender(face_roi, gender_net, age_net)
                
                # Saç rengi tespiti - burada basit bir kontrol kullanıyoruz
                # Bu basit bir varsayım olup daha gelişmiş bir model kullanılabilir.
                hair_color = "Black" if np.mean(face_roi) < 100 else "Blonde"
                
                # Videonun sağ üst köşesine cinsiyet, yaş ve saç rengini ekle
                text = f"Age: {age}, Gender: {gender}, Hair: {hair_color}"
                draw_text(frame, text, (10, 40))
        
        # Çıkış videosuna yaz
        out.write(frame)
    
    # Kaynakları serbest bırak
    cap.release()
    out.release()
    face_detection.close()
    print("İşlem tamamlandı. Çıkış video dosyası kaydedildi.")

def main():
    # PY klasöründeki videoları bul
    py_folder = "C:/Users/BRAKENUS/Desktop/PY"
    videos = get_videos_from_folder(py_folder)
    
    if not videos:
        print("PY klasöründe işlenecek video bulunamadı.")
        return
    
    # Tespit edilen tüm videoları sırayla işleme
    for video in videos:
        print(f"İşleniyor: {video}")
        output_path = os.path.join(py_folder, f"processed_{os.path.basename(video)}")
        process_video(video, output_path)

if __name__ == "__main__":
    main()
