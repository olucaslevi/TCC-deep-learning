import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

# Carregar o modelo treinado
model = keras.models.load_model('mnist_cnn_model.h5')

# Função para pré-processar uma imagem única
def preprocess_image(img):
    # Redimensionar a imagem para o tamanho esperado (28x28)
    img = cv2.resize(img, (28, 28))
    img_array = np.expand_dims(img, axis=0)
    # Normalizar os valores de pixel
    img_array = img_array / 255.0
    return img_array

# Inicializar a webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Detectar dígitos na imagem da webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Apenas considerar contornos de tamanho mínimo
            digit_region = gray[y:y+h, x:x+w]
            digit_image = cv2.resize(digit_region, (28, 28))
            input_image = preprocess_image(digit_image)
            predictions = model.predict(input_image)
            predicted_class = np.argmax(predictions)
            
            if predicted_class >= 0 and predicted_class <= 9:
                # Desenhar um retângulo ao redor do dígito reconhecido
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Exibir o número reconhecido acima do retângulo
                cv2.putText(frame, str(predicted_class), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Exibir a imagem da webcam
    cv2.imshow('Webcam', frame)

    # Sair do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a webcam e fechar a janela
cap.release()
cv2.destroyAllWindows()
