from tensorflow.keras.models import load_model

try:
    model = load_model('model/final_gi_model.h5')
    print("✅ Model loaded!")
    print("Input shape:", model.input_shape)
except Exception as e:
    print("❌ Failed:", e)