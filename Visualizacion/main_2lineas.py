# Importar librerias
from ultralytics import YOLO
import cv2
import time
from collections import deque

# Leer nuestro modelo
model = YOLO("best_Renato.pt")

# Realizar videocaptura
cap = cv2.VideoCapture(0)

# Inicializar variables para el seguimiento del texto
texto_mostrado = deque(maxlen=2)  # Usar deque para mantener un máximo de 3 líneas de texto
ultimo_texto_mostrado = ""
ultimo_tiempo_actualizacion = time.time()

# Mapeo de clases a etiquetas
clases_a_etiquetas = {
    0: "abrazar",
    1: "dentro",
    2: "te",
    3: "universidad",
    4: "caminar",
    5: "ensenar",
    6: "Peru",
    7: "mio",
    8: "jugar",
}

# Bucle
while True:
    
    # Leer fotogramas
    ret, frame = cap.read()
    
    # Leer resultados
    resultados = model.predict(frame, imgsz=640, conf=0.7)
    
    # Mostrar resultados en la ventana
    anotaciones = resultados[0].plot()
    
    for result in resultados:
        boxes = result.boxes  # Boxes object for bbox outputs

        # Obtener las clases como lista de Python
        clases_detectadas = boxes.cls.cpu().numpy().tolist()
        clase_detectada = None

        if clases_detectadas:
            clase_detectada = clases_detectadas[0]

        # Obtener la etiqueta correspondiente
        if clase_detectada is not None:
            nueva_etiqueta = clases_a_etiquetas.get(clase_detectada, "Ninguna palabra detectada")
            if nueva_etiqueta != ultimo_texto_mostrado:
                if texto_mostrado:
                    # Unir las palabras de la línea actual en un solo string
                    linea_actual = " ".join(texto_mostrado[0])
                    # Calcular el ancho del texto actual
                    ancho_texto_actual, _ = cv2.getTextSize(linea_actual, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    # Calcular el ancho de la nueva palabra
                    ancho_nueva_palabra, _ = cv2.getTextSize(nueva_etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    # Si la nueva palabra cabe en la línea actual, agregarla (La línea actual más la nueva palabra debe caber en el ancho de la ventana)
                    if ancho_texto_actual[0] + ancho_nueva_palabra[0] + 20 < anotaciones.shape[1]:  # 10 es un margen
                        texto_mostrado[0].append(nueva_etiqueta)
                    else:
                        # Si la nueva palabra no cabe en la línea actual, agregarla en una nueva línea
                        texto_mostrado.appendleft([nueva_etiqueta])
                else:
                    texto_mostrado.appendleft([nueva_etiqueta])
                ultimo_texto_mostrado = nueva_etiqueta
                ultimo_tiempo_actualizacion = time.time()

    # Si han pasado 5 segundos desde la última actualización, reiniciar el texto
    if time.time() - ultimo_tiempo_actualizacion > 5:
        texto_mostrado.clear()  # Limpiar todas las líneas de texto

    # Obtener el tamaño de la imagen
    height, width, _ = anotaciones.shape

    # Calcular la altura del texto
    text_height = cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][1]

    # Dibujar cada línea de texto
    y = height - 10  # Iniciar en la parte inferior de la ventana
    print(texto_mostrado)
    for line in texto_mostrado:
        # Ajustar la posición vertical de la línea
        y -= text_height + 5  # Restar el tamaño del texto más un pequeño margen entre líneas
        # Crear una cadena con las palabras de la línea
        line_str = " ".join(line)
        # Dibujar sombreado del texto (en negro)
        cv2.putText(anotaciones, line_str, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        # Mostrar el texto en la imagen con anotaciones (en la parte inferior)
        cv2.putText(anotaciones, line_str, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mostrar la imagen con anotaciones
    cv2.imshow("Detección y segmentación", anotaciones)

    # Cerrar programa al presionar 'Esc'
    if cv2.waitKey(1) == 27:
        break

# Liberar la captura de video y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
