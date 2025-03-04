import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load model TFLite dengan Edge TPU delegate
interpreter = tflite.Interpreter(
    model_path="yolo11n_full_integer_quant_edgetpu.tflite",
    experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")]
)
interpreter.allocate_tensors()

# Ambil detail input & output tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Dapatkan ukuran input model (biasanya 320x320 atau 640x640)
input_shape = input_details[0]['shape']
input_size = (input_shape[1], input_shape[2])  # (height, width)

# Buka webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame ke ukuran input model
    img = cv2.resize(frame, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ubah ke RGB
    img = np.expand_dims(img, axis=0)  # Tambah batch dimension
    img = np.uint8(img)  # Konversi ke uint8 (karena model quantized)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Jalankan inferensi
    interpreter.invoke()

    # Ambil output dari model
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Parsing hasil deteksi (format output tergantung pada model)
    for obj in output_data[0]:  
        x1, y1, x2, y2, conf, class_id = obj[:6]  # Ambil koordinat dan skor
        if conf > 0.5:  # Tampilkan hanya yang confidence > 50%
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Konversi ke int
            
            # Gambar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Tambahkan label
            label = f"Class {int(class_id)}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    # Tampilkan hasil di jendela
    cv2.imshow("Edge TPU YOLO Detection", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()
