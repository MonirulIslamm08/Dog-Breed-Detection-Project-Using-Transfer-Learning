from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pickle
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure upload directory exists

# Load model and class labels
try:
    model = load_model('dog_breed_detector.keras')
    with open('class_labels.pkl', 'rb') as f:
        class_labels = pickle.load(f)
except Exception as e:
    print(f"Error loading model or class labels: {e}")
    # You could add fallback behavior here

# Optional breed info (you can replace or extend this)
breed_descriptions = {
    'beagle': "Beagles are friendly, curious, and merry dogs with a wonderful sense of smell. They're medium-sized with a sturdy build, known for their long droopy ears and expressive eyes. Originally bred for hunting, they have a strong tracking instinct and love to follow their nose.",
    'bulldog': "Bulldogs are gentle, loyal, and courageous companions. With their distinctive pushed-in nose, wrinkled face, and sturdy build, they're easily recognizable. Despite their tough appearance, they're typically docile and great with families.",
    'dalmatian': "Dalmatians are known for their distinctive spots and energetic nature. These athletic dogs were historically used as carriage dogs and firehouse mascots. They're intelligent, playful, and make excellent companions for active families.",
    'german-shepherd': "German Shepherds are intelligent, versatile working dogs known for their loyalty and courage. They excel in various roles including police, military, and service work. With proper training and socialization, they make wonderful family protectors.",
    'husky': "Huskies are known for their endurance, striking appearance, and wolf-like features. Bred as sled dogs in cold climates, they have a thick double coat and high energy levels. They're known for being independent, friendly, and somewhat mischievous.",
    'labrador-retriever': "Labrador Retrievers are friendly, outgoing, and high-spirited companions. They're America's most popular dog breed for good reason - they're excellent with children, eager to please, and relatively easy to train. They excel as service dogs and family pets.",
    'poodle': "Poodles are intelligent, active, and elegant dogs available in three sizes (standard, miniature, and toy). Despite their sophisticated appearance, they were originally bred as water retrievers. They're highly trainable and known for their hypoallergenic coat.",
    'rottweiler': "Rottweilers are confident, powerful protectors with a natural guarding instinct. When properly trained and socialized, they're loving and loyal family companions. They're strong working dogs that excel in roles requiring strength and intelligence."
}

def predict_dog_breed(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_label = class_labels[predicted_index]
        confidence = float(np.max(predictions))
        return predicted_label, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error processing image", 0.0

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            # Create unique filename to prevent overwriting
            filename = f"{uuid.uuid4()}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                label, confidence = predict_dog_breed(filepath)
                confidence_formatted = f"{confidence * 100:.1f}"
                description = breed_descriptions.get(label, "No additional information available for this breed.")
                
                return render_template('index.html',
                                    prediction=label.replace('-', ' ').title(),
                                    confidence=confidence_formatted,
                                    image_path=filepath,
                                    breed_info=description)
            except Exception as e:
                return render_template('index.html', 
                                    prediction="Error",
                                    confidence="0.0", 
                                    image_path=filepath,
                                    breed_info=f"An error occurred during processing: {str(e)}")
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
