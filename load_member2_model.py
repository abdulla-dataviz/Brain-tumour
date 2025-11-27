from tensorflow.keras.models import load_model

def load_my_model():
    model = load_model("models/best_model.h5")
    return model
