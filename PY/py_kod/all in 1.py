import os
import cv2

def find_file_in_directory(directory, filename):
    """Belirtilen klasörde dosyayı arar ve tam yolunu döndürür."""
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None

def initialize_models(base_dir):
    """Yaş ve cinsiyet tahmini modellerini başlatır."""
    files_dir = os.path.join(base_dir, "Files")

    # Dosya adlarını tanımla
    age_prototxt = "age_deploy.prototxt"
    age_caffemodel = "age_net.caffemodel"
    gender_prototxt = "deploy_gender.prototxt"
    gender_caffemodel = "gender_net.caffemodel"
    
    # Dosyaları arama
    age_prototxt_path = find_file_in_directory(files_dir, age_prototxt)
    age_caffemodel_path = find_file_in_directory(files_dir, age_caffemodel)
    gender_prototxt_path = find_file_in_directory(files_dir, gender_prototxt)
    gender_caffemodel_path = find_file_in_directory(files_dir, gender_caffemodel)
    
    if not age_prototxt_path or not age_caffemodel_path or not gender_prototxt_path or not gender_caffemodel_path:
        raise FileNotFoundError("Model dosyaları bulunamadı!")
    
    # Yaş modelini yükle
    age_net = cv2.dnn.readNetFromCaffe(age_prototxt_path, age_caffemodel_path)
    
    # Cinsiyet modelini yükle
    gender_net = cv2.dnn.readNetFromCaffe(gender_prototxt_path, gender_caffemodel_path)
    
    return age_net, gender_net

def find_video_to_process(base_dir):
    """Video klasöründe işlenecek dosyayı bulur."""
    video_dir = os.path.join(base_dir, "Video")
    for file in os.listdir(video_dir):
        if file.endswith(".mp4") and "yüz" in file.lower():
            return os.path.join(video_dir, file)
    return None

def process_video(video_path, age_net, gender_net):
    """Videoyu işleyip cinsiyet ve yaş tahmini uygular."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Video açılamadı: {video_path}")
    
    # Video özelliklerini al
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Çıkış video yazıcısını ayarla
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_dir = "C:/Users/BRAKENUS/Desktop/PY/Çıktılar"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"processed_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Yüz algılama için OpenCV'nin önceden eğitilmiş modelini kullan
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Grayscale (gri tonlama) dönüşümü
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Yüz tespiti
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Yüz bölgesini kırp ve yaş tahmini yap
            face = frame[y:y+h, x:x+w]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            
            # Yaş tahmini yap
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age_ranges = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
            age = age_ranges[age_preds[0].argmax()]
            
            # Cinsiyet tahmini yap
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = "Female" if gender_preds[0].argmax() == 0 else "Male"
            
            # Yüzün etrafına dikdörtgen çiz ve tahminleri ekle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {age}, Gender: {gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # İşlenmiş kareyi çıktı videosuna yaz
        out.write(frame)

    # Kaynakları serbest bırak
    cap.release()
    out.release()
    print(f"Video işleme tamamlandı. Çıktı: {output_path}")

def main():
    base_dir = "C:/Users/BRAKENUS/Desktop/PY"
    
    try:
        # Modelleri başlat
        age_net, gender_net = initialize_models(base_dir)
        print("Yaş ve cinsiyet modelleri başarıyla yüklendi.")
        
        # İşlenecek videoyu bul
        video_path = find_video_to_process(base_dir)
        if not video_path:
            raise FileNotFoundError("PY klasöründe işlenecek video bulunamadı.")
        print(f"İşlenecek video: {video_path}")
        
        # Videoyu işle ve çıktıyı kaydet
        output_dir = os.path.join(base_dir, "Çıktılar")
        process_video(video_path, age_net, gender_net)
        print(f"Video işleme tamamlandı ve şu klasöre kaydedildi: {output_dir}")
    
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    main()
