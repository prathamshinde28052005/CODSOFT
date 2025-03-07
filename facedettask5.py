import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class FaceDetectionRecognition:
    def __init__(self, recognition_enabled=False, model_type='haar'):
        """
        Initialize the Face Detection and Recognition system
        
        Args:
            recognition_enabled (bool): Whether to enable face recognition
            model_type (str): Type of face detection model ('haar', 'dnn', 'mtcnn')
        """
        self.recognition_enabled = recognition_enabled
        self.model_type = model_type
        self.face_detector = None
        self.face_recognizer = None
        self.label_encoder = LabelEncoder()
        
        # Initialize face detector
        self.initialize_face_detector()
        
        # Initialize face recognizer if enabled
        if recognition_enabled:
            self.initialize_face_recognizer()
    
    def initialize_face_detector(self):
        """Initialize the selected face detection model"""
        if self.model_type == 'haar':
            # Haar Cascade classifier
            self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        elif self.model_type == 'dnn':
            # Deep Neural Network based detector
            model_file = "models/res10_300x300_ssd_iter_140000.caffemodel"
            config_file = "models/deploy.prototxt.txt"
            
            # Check if model files exist, if not, provide download instructions
            if not os.path.exists(model_file) or not os.path.exists(config_file):
                os.makedirs("models", exist_ok=True)
                print("DNN model files not found. Please download from:")
                print("https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector")
                print(f"and place them in {os.path.abspath('models/')}")
            
            self.face_detector = cv2.dnn.readNetFromCaffe(config_file, model_file)
        
        elif self.model_type == 'mtcnn':
            # Import MTCNN here to make it optional
            try:
                from mtcnn.mtcnn import MTCNN
                self.face_detector = MTCNN()
            except ImportError:
                print("MTCNN not installed. Please install with: pip install mtcnn tensorflow")
                self.model_type = 'haar'
                self.initialize_face_detector()
    
    def initialize_face_recognizer(self):
        """Initialize the face recognition model (Siamese Network)"""
        # Define the Siamese Network architecture
        input_shape = (96, 96, 3)  # Standard face size after preprocessing
        
        # Base network for feature extraction
        base_network = Sequential([
            Conv2D(64, (10, 10), activation='relu', input_shape=input_shape),
            MaxPooling2D(),
            Conv2D(128, (7, 7), activation='relu'),
            MaxPooling2D(),
            Conv2D(128, (4, 4), activation='relu'),
            MaxPooling2D(),
            Conv2D(256, (4, 4), activation='relu'),
            Flatten(),
            Dense(4096, activation='sigmoid')
        ])
        
        # Create the Siamese Network
        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)
        
        # Feature vectors for both inputs
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        
        # L1 distance layer
        distance = Lambda(lambda x: tf.keras.backend.abs(x[0] - x[1]))([processed_a, processed_b])
        
        # Final layer for similarity prediction
        prediction = Dense(1, activation='sigmoid')(distance)
        
        # Define the Siamese model
        self.face_recognizer = Model(inputs=[input_a, input_b], outputs=prediction)
        self.face_recognizer.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00006), metrics=['accuracy'])
        
        # Also create a model for feature extraction from a single image (for prediction)
        self.feature_extractor = Model(inputs=base_network.input, outputs=base_network.output)
    
    def detect_faces(self, image):
        """
        Detect faces in an image
        
        Args:
            image: Input image (BGR format for OpenCV)
            
        Returns:
            List of detected faces in the format [x, y, width, height]
        """
        faces = []
        img_height, img_width = image.shape[:2]
        
        if self.model_type == 'haar':
            # Convert to grayscale for Haar Cascade
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            detections = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Format: [x, y, width, height]
            faces = detections.tolist() if len(detections) > 0 else []
        
        elif self.model_type == 'dnn':
            # Prepare image for deep neural network
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 
                1.0, 
                (300, 300), 
                (104.0, 177.0, 123.0)
            )
            
            # Detect faces
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()
            
            # Extract face bounding boxes
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter detections by confidence
                if confidence > 0.5:
                    # Get bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([img_width, img_height, img_width, img_height])
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Convert to [x, y, width, height] format
                    faces.append([x1, y1, x2 - x1, y2 - y1])
        
        elif self.model_type == 'mtcnn':
            # MTCNN already works with RGB, so convert from BGR
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detections = self.face_detector.detect_faces(rgb_image)
            
            # Extract face bounding boxes
            for detection in detections:
                x, y, width, height = detection['box']
                faces.append([x, y, width, height])
        
        return faces
    
    def preprocess_face(self, image, face_box, target_size=(96, 96)):
        """
        Extract and preprocess a face from an image
        
        Args:
            image: Input image
            face_box: Face bounding box [x, y, width, height]
            target_size: Size to resize face to
            
        Returns:
            Preprocessed face image
        """
        x, y, width, height = face_box
        
        # Extract face with some margin
        margin = int(min(width, height) * 0.1)
        x = max(0, x - margin)
        y = max(0, y - margin)
        width = min(image.shape[1] - x, width + 2 * margin)
        height = min(image.shape[0] - y, height + 2 * margin)
        
        # Extract face region
        face = image[y:y+height, x:x+width]
        
        # Convert to RGB (for models expecting RGB input)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        face_resized = cv2.resize(face_rgb, target_size)
        
        # Normalize pixel values
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        return face_normalized
    
    def train_recognizer(self, dataset_path, epochs=20, batch_size=32):
        """
        Train the face recognition model on a dataset of face images
        
        Args:
            dataset_path: Path to dataset with structure: dataset_path/person_name/image.jpg
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if not self.recognition_enabled:
            print("Face recognition is not enabled. Initialize with recognition_enabled=True")
            return None
        
        # Load dataset
        X = []  # Images
        y = []  # Labels (person names)
        
        # Process each person's directory
        for person_name in os.listdir(dataset_path):
            person_dir = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            # Process each image in person's directory
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                # Read and detect face in image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                    
                faces = self.detect_faces(image)
                if not faces:
                    print(f"No face detected in {image_path}")
                    continue
                
                # Use the first detected face
                face_box = faces[0]
                face_img = self.preprocess_face(image, face_box)
                
                X.append(face_img)
                y.append(person_name)
        
        if not X:
            print("No valid face images found in the dataset")
            return None
        
        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Create pairs for Siamese network training
        pairs, labels = self._create_pairs(X, y_encoded)
        
        # Split data into training and validation sets
        pairs_train, pairs_val, labels_train, labels_val = train_test_split(
            pairs, labels, test_size=0.2, random_state=42
        )
        
        # Data augmentation for training
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        
        # Train the model
        history = self.face_recognizer.fit(
            [pairs_train[:, 0], pairs_train[:, 1]], labels_train,
            validation_data=([pairs_val[:, 0], pairs_val[:, 1]], labels_val),
            batch_size=batch_size,
            epochs=epochs
        )
        
        # Save the model and label encoder
        self.face_recognizer.save("models/face_recognition_model.h5")
        np.save("models/label_encoder.npy", self.label_encoder.classes_)
        
        return history
    
    def _create_pairs(self, X, y):
        """
        Create positive and negative pairs for Siamese network training
        
        Args:
            X: Face images
            y: Encoded labels
            
        Returns:
            pairs, labels (1 for same person, 0 for different people)
        """
        num_classes = len(np.unique(y))
        digit_indices = [np.where(y == i)[0] for i in range(num_classes)]
        
        pairs = []
        labels = []
        
        # For each class (person)
        for d in range(num_classes):
            # For each image of the person
            for i in range(len(digit_indices[d]) - 1):
                # Create positive pair (same person)
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                pairs += [[X[z1], X[z2]]]
                labels += [1]
                
                # Create negative pair (different people)
                inc = (d + 1) % num_classes  # Select the next class
                z1, z2 = digit_indices[d][i], digit_indices[inc][i % len(digit_indices[inc])]
                pairs += [[X[z1], X[z2]]]
                labels += [0]
        
        return np.array(pairs), np.array(labels)
    
    def recognize_face(self, face_img, reference_faces, reference_labels, threshold=0.5):
        """
        Recognize a face by comparing it with reference faces
        
        Args:
            face_img: Preprocessed face image
            reference_faces: List of preprocessed reference face images
            reference_labels: Corresponding labels for reference faces
            threshold: Similarity threshold
            
        Returns:
            Predicted label and confidence score
        """
        if not self.recognition_enabled:
            return None, 0
        
        # Extract features of the input face
        face_features = self.feature_extractor.predict(np.expand_dims(face_img, axis=0))[0]
        
        best_match = None
        best_score = 0
        
        # Compare with each reference face
        for ref_face, ref_label in zip(reference_faces, reference_labels):
            # Extract features of the reference face
            ref_features = self.feature_extractor.predict(np.expand_dims(ref_face, axis=0))[0]
            
            # Calculate similarity (1 - Manhattan distance normalized to [0, 1])
            similarity = 1.0 - np.sum(np.abs(face_features - ref_features)) / len(face_features)
            
            # Update best match
            if similarity > best_score:
                best_score = similarity
                best_match = ref_label
        
        # Apply threshold
        if best_score < threshold:
            return "Unknown", best_score
        
        return best_match, best_score
    
    def load_recognizer(self, model_path="models/face_recognition_model.h5", 
                        encoder_path="models/label_encoder.npy"):
        """
        Load a pre-trained face recognition model
        
        Args:
            model_path: Path to saved model
            encoder_path: Path to saved label encoder
            
        Returns:
            True if successful, False otherwise
        """
        if not self.recognition_enabled:
            print("Face recognition is not enabled. Initialize with recognition_enabled=True")
            return False
        
        try:
            # Load model
            self.face_recognizer = tf.keras.models.load_model(model_path)
            
            # Extract feature extractor model
            base_network = self.face_recognizer.layers[2]
            self.feature_extractor = Model(inputs=base_network.input, outputs=base_network.output)
            
            # Load label encoder
            self.label_encoder.classes_ = np.load(encoder_path)
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def process_video(self, video_source=0, output_path=None, display=True, recognition=False):
        """
        Process video for face detection and optional recognition
        
        Args:
            video_source: Camera index or video file path
            output_path: Path to save the processed video
            display: Whether to display the video
            recognition: Whether to perform face recognition
            
        Returns:
            None
        """
        # Open video capture
        cap = cv2.VideoCapture(video_source)
        
        # Check if opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer if output path is specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Load reference faces for recognition if enabled
        reference_faces = []
        reference_labels = []
        if recognition and self.recognition_enabled:
            print("Loading reference faces for recognition...")
            # This would typically load a small gallery of faces for comparison
            # For simplicity, we'll assume these are loaded from somewhere
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                
                # Break if end of video
                if not ret:
                    break
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process each detected face
                for face_box in faces:
                    x, y, width, height = face_box
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    
                    # Perform recognition if enabled
                    if recognition and self.recognition_enabled:
                        # Preprocess face
                        face_img = self.preprocess_face(frame, face_box)
                        
                        # Recognize face
                        name, confidence = self.recognize_face(
                            face_img, reference_faces, reference_labels
                        )
                        
                        # Display name and confidence
                        text = f"{name} ({confidence:.2f})"
                        cv2.putText(frame, text, (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Write frame to output video
                if out:
                    out.write(frame)
                
                # Display frame
                if display:
                    cv2.imshow('Face Detection', frame)
                    
                    # Exit on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            # Release resources
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

# Example usage
def main():
    """Example usage of the FaceDetectionRecognition class"""
    # Initialize with different detector types
    
    # 1. Simple face detection with Haar cascade
    print("Running Haar cascade face detection demo...")
    haar_detector = FaceDetectionRecognition(recognition_enabled=False, model_type='haar')
    
    # Process webcam video
    haar_detector.process_video(video_source=0, display=True, recognition=False)
    
    # 2. Face detection with DNN (better accuracy)
    print("Running DNN face detection demo...")
    dnn_detector = FaceDetectionRecognition(recognition_enabled=False, model_type='dnn')
    
    # Process a video file
    video_path = "path/to/video.mp4"  # Replace with actual path
    if os.path.exists(video_path):
        dnn_detector.process_video(
            video_source=video_path,
            output_path="output_dnn.avi",
            display=True
        )
    
    # 3. Face detection and recognition
    print("Running face recognition demo...")
    face_recognizer = FaceDetectionRecognition(recognition_enabled=True, model_type='haar')
    
    # Train recognizer on dataset (if available)
    dataset_path = "path/to/faces"  # Replace with actual path
    if os.path.exists(dataset_path):
        print("Training face recognizer...")
        history = face_recognizer.train_recognizer(dataset_path, epochs=10)
        
        # Plot training history
        if history:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()
    
    # Or load pre-trained recognizer
    model_path = "models/face_recognition_model.h5"
    if os.path.exists(model_path):
        print("Loading pre-trained face recognizer...")
        face_recognizer.load_recognizer()
        
        # Process webcam with recognition
        face_recognizer.process_video(
            video_source=0,
            output_path="output_recognition.avi",
            display=True,
            recognition=True
        )

if __name__ == "__main__":
    main()