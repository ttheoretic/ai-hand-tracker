import cv2
import mediapipe as mp

# Initialisiere Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Starte die Webcam mit AVFoundation-Backend
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden")
    exit()

while True:
    # Lies ein Bild von der Webcam
    success, image = cap.read()
    
    if not success:
        print("Fehler beim Lesen des Kamerabildes")
        break
    
    # Konvertiere das Bild von BGR zu RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Verarbeite das Bild mit Mediapipe Hands
    results = hands.process(image_rgb)
    
    # Wenn Hände erkannt wurden
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Zeichne die Landmarken auf das Bild
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Hole die Koordinaten der Fingerspitzen
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Zeichne einen Kreis auf die Fingerspitzen (IDs 4, 8, 12, 16, 20)
                if id in [4, 8, 12, 16, 20]:
                    cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
    
    # Zeige das Bild an
    cv2.imshow("Hand Tracking", image)
    
    # Beende das Programm, wenn 'q' gedrückt wird
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Räume auf
cap.release()
cv2.destroyAllWindows()
