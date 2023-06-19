# Function to detect eyes in a face picture
def eyes_recognition(image):

  # Prepare image for eyes detection
  color = cv2.imread(image)
  gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

  # Extract landmarks coordinates
  landmarks = extract_face_landmarks(gray)

  # Calculate a margin around the eye
  extractmarge = int(len(gray)*0.05)

  # Left eye maximal coordinates
  lx1 = landmarks[36][0]
  lx2 = landmarks[39][0]
  ly1 = landmarks[37][1]
  ly2 = landmarks[40][1]
  lefteye = color[ly1 - extractmarge : ly2 + extractmarge, lx1 - extractmarge : lx2 + extractmarge]

  # Right eye maximal coordinates
  rx1 = landmarks[42][0]
  rx2 = landmarks[45][0]
  ry1 = landmarks[43][1]
  ry2 = landmarks[46][1]
  righteye = color[ry1 - extractmarge : ry2 + extractmarge, rx1 - extractmarge : rx2 + extractmarge]

  # Return eyes images
  return lefteye, righteye


# Function to preprocess eye informations extract with eye_detection function before launching the prediction
def eye_preprocess(eye):

  # Resize your image to fit model entry
  resize = tf.image.resize(
    eye,
    size = (52, 52),
    method = tf.image.ResizeMethod.BILINEAR
  )

  # Switch to grayscale
  grayscale = tf.image.rgb_to_grayscale(
      resize
  )

  # Normalize your data
  norm = grayscale / 255

  # Add one dimension to fit model entry
  final = tf.expand_dims(
      norm, axis = 0
  )

  # Return the final image to make your prediction
  return final

def prediction(lefteye, righteye, model):

  class_labels = ["close", "open"]

  # Predict and return predictions
  # For lefteye
  preds_left = model.predict(lefteye)
  pred_left = np.argmax(preds_left, axis = 1)
  # For righteye
  preds_right = model.predict(righteye)
  pred_right = np.argmax(preds_right, axis = 1)

  if pred_left == pred_right:
    state = class_labels[pred_left[0]]
  else:
    state = "wink"

  return state