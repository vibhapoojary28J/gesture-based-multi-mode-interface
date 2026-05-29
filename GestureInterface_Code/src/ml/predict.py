def _extract_features(landmarks):
    """
    Convert normalized landmark data into a flat feature vector.
    Supports landmarks in the form (id, x, y) or (x, y).
    """
    if not landmarks:
        return []

    features = []
    for lm in landmarks:
        if len(lm) == 3:
            _, x, y = lm
        elif len(lm) >= 2:
            x, y = lm[0], lm[1]
        else:
            continue
        features.extend([x, y])
    return features


def predict_gesture(landmarks, model):
    """
    Predict a gesture label from landmarks using a trained model.
    Returns 'UNKNOWN' if prediction cannot be made.
    """
    if model is None or not landmarks:
        return 'UNKNOWN'

    features = _extract_features(landmarks)
    if not features:
        return 'UNKNOWN'

    try:
        prediction = model.predict([features])
        if isinstance(prediction, (list, tuple)):
            return str(prediction[0])
        return str(prediction)
    except Exception as e:
        print(f"ML prediction failed: {e}")
        return 'UNKNOWN'
