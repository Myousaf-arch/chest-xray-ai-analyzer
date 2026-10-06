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

def predict_single_image(user_input):
    user_input = user_input.replace('"', '').replace("'", "")
    
    try:
        if user_input.startswith("http://") or user_input.startswith("https://"):
            response = requests.get(user_input, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
            img = img.resize((150, 150))
        else:
            if not os.path.exists(user_input):
                print(f"\nERROR: File path '{user_input}' nahi mila!\n")
                return
            img = image.load_img(user_input, target_size=(150, 150))

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # FINAL LOGIC: 0 = Normal, 1 = Pneumonia
        print("\n" + "="*45)
        if prediction > 0.5:
            confidence = prediction * 100
            print(f"RESULT: PNEUMONIA DETECTED! ({confidence:.2f}% Confidence)")
        else:
            confidence = (1 - prediction) * 100
            print(f"RESULT: NORMAL X-RAY ({confidence:.2f}% Confidence)")
        print("="*45 + "\n")

    except Exception as e:
        print(f"\nERROR: Image process nahi ho saki ({e})\n")

def main():
    print("--------------------------------------------------")
    print(" Chest X-Ray Pneumonia Predictor (Continuous Loop)")
    print(" Exit karne ke liye 'exit' ya 'q' likhein.")
    print("--------------------------------------------------\n")

    while True:
        # NAYA PROMPT YAHAN HAI
        user_input = input("Enter Image Local Path / URL (ya 'q' to quit): ").strip()
        
        if user_input.lower() in ['exit', 'q', 'quit', 'stop']:
            print("\nProgram band ho raha hai. Goodbye!")
            break
        
        if not user_input:
            continue
            
        predict_single_image(user_input)

if __name__ == "__main__":
    main()