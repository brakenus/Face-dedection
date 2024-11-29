import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import sys
def get_videos_from_folder(folder_path):
    # Belirtilen klasördeki tüm mp4 dosyalarını listele
    videos = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.mp4')]
    return videos

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Videodaki tüm yüzleri bulanıklaştıran script. Yüzler hareket ettikçe bulanıklık takip eder.',
        epilog='Örnek kullanım: python blur_faces_in_video.py --input_video /path/to/input.mp4 --output_video /path/to/output.mp4'
    )
    
    parser.add_argument('--input_video', type=str, required=True, help='Giriş video dosyasının yolu')
    parser.add_argument('--output_video', type=str, required=True, help='Çıkış video dosyasının kaydedileceği yol')
    parser.add_argument('--blur_level', type=int, default=30, help='Bulanıklaştırma seviyesini belirler (default: 30)')
    
    return parser.parse_args()

def initialize_mediapipe():
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    return face_detection

def blur_face_regions(frame, detections, blur_level):
    image_height, image_width = frame.shape[:2]
    for detection in detections:
        # Yüzün bounding box koordinatlarını al
        bboxC = detection.location_data.relative_bounding_box
        x_min = int(bboxC.xmin * image_width)
        y_min = int(bboxC.ymin * image_height)
        box_width = int(bboxC.width * image_width)
        box_height = int(bboxC.height * image_height)
        
        # Bounding box'un frame sınırları dışına çıkmasını önle
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_min + box_width, image_width)
        y_max = min(y_min + box_height, image_height)
        
        # Yüz bölgesini kırp
        face_roi = frame[y_min:y_max, x_min:x_max]
        
        # Bulanıklaştırma
        if face_roi.size != 0:
            # Bulanıklık seviyesini tek bir sayıya ayarla
            blur_level = max(1, blur_level // 2 * 2 + 1)  # Çiftten teke çevir
            blurred_face = cv2.GaussianBlur(face_roi, (blur_level, blur_level), 0)

            frame[y_min:y_max, x_min:x_max] = blurred_face
    return frame

def process_video(input_path, output_path, blur_level):
    if not os.path.exists(input_path):
        print(f"Giriş video dosyası bulunamadı: {input_path}")
        sys.exit(1)
    
    # Video yakalama
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Video açılamadı: {input_path}")
        sys.exit(1)
    
    # Video özelliklerini al
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Çıkış formatı (mp4)
    
    # Video yazıcıyı başlat
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Mediapipe'yi başlat
    face_detection = initialize_mediapipe()
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    
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
            # Yüzleri bulanıklaştır
            frame = blur_face_regions(frame, results.detections, blur_level)
        
        # Çıkış videosuna yaz
        out.write(frame)
        
        current_frame += 1
        if current_frame % 30 == 0:
            print(f"İşlenen kare: {current_frame}/{frame_count}")
    
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
        output_path = os.path.join(py_folder, f"işlenmiş_{os.path.basename(video)}")
        process_video(video, output_path, blur_level=30)


if __name__ == "__main__":
    main()
