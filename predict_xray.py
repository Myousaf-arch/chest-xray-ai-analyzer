import os
import io
import requests
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image

print("--- Loading Saved Chest X-Ray Model ---")
model = tf.keras.models.load_model("chest_xray_model.keras")
print("Model Ready!\n")

def predict_from_input():
    user_input = input("Enter Image Local Path OR Google Image Link (URL): ").strip()
    user_input = user_input.replace('"', '').replace("'", "")
    
    try:
        if user_input.startswith("http://") or user_input.startswith("https://"):
            response = requests.get(user_input, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
            img = img.resize((150, 150))
        else:
            if not os.path.exists(user_input):
                print(f"\nERROR: File path '{user_input}' nahi mila!")
                return
            img = image.load_img(user_input, target_size=(150, 150))

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        prediction = model.predict(img_array)[0][0]
        print(f"\nRaw Model Prediction Value: {prediction:.4f}")

        # LOGIC REVERSED FIX:
        # Agar Raw Value < 0.5 ho to Pneumonia, warna Normal
        print("\n" + "="*40)
        if prediction < 0.5:
            confidence = (1 - prediction) * 100
            print(f"RESULT: PNEUMONIA DETECTED! ({confidence:.2f}% Confidence)")
        else:
            confidence = prediction * 100
            print(f"RESULT: NORMAL X-RAY ({confidence:.2f}% Confidence)")
        print("="*40 + "\n")

    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    predict_from_input()