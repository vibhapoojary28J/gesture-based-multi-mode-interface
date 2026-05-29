import pickle
from pathlib import Path


def load_model(model_path=None):
    """
    Load a trained gesture model from disk.
    Returns a deserialized model instance or None on failure.
    """
    if model_path is None:
        base_dir = Path(__file__).resolve().parents[2]
        model_path = base_dir / "models" / "gesture_model.pkl"

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
    return None
