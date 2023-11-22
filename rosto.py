import cv2

# Carrega a imagem
image = cv2.imread('path/to/image.jpg')

# Converte a imagem para escala de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Carrega o classificador de detecção de faces
face_cascade = cv2.CascadeClassifier('path/to/haarcascade_frontalface_default.xml')

# Detecta as faces na imagem
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Desenha um retângulo em volta de cada face detectada
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Exibe a imagem com as faces detectadas
cv2.imshow('Faces detectadas', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
