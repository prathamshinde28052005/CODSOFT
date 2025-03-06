import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

# 1. Image Feature Extraction Model
def build_feature_extractor():
    # Use ResNet50 as base model
    base_model = ResNet50(weights='imagenet')
    # Remove the classification layer
    feature_model = Model(inputs=base_model.input, 
                         outputs=base_model.layers[-2].output)
    return feature_model

# 2. Caption Generation Model (CNN + LSTM)
def build_caption_model(vocab_size, max_length):
    # Image feature input
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence feature input
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # Decoder
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image

# 1. Image Preprocessing
def preprocess_image(img_path, target_size=(224, 224)):
    # Load the image
    img = image.load_img(img_path, target_size=target_size)
    # Convert to array
    x = image.img_to_array(img)
    # Expand dimensions
    x = np.expand_dims(x, axis=0)
    # Preprocess for ResNet
    x = tf.keras.applications.resnet50.preprocess_input(x)
    return x

def extract_features(feature_extractor, image_path):
    image = preprocess_image(image_path)
    feature = feature_extractor.predict(image, verbose=0)
    return feature

# 2. Caption Preprocessing
def create_tokenizer(captions):
    # Flatten all caption lists
    all_captions = []
    for img_captions in captions.values():
        all_captions.extend(img_captions)
    
    # Create tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    
    return tokenizer

# 3. Training Loop
def train_caption_model(model, features, captions, tokenizer, max_length, vocab_size, epochs=10, batch_size=32):
    # Prepare training data
    X1, X2, y = list(), list(), list()
    
    # For each image-caption pair
    for img_id, caption_list in captions.items():
        # Get image features
        feature = features[img_id]
        
        # Process each caption
        for caption in caption_list:
            # Encode the sequence
            seq = tokenizer.texts_to_sequences([caption])[0]
            
            # Create training samples for each word
            for i in range(1, len(seq)):
                # Split into input and output parts
                in_seq, out_seq = seq[:i], seq[i]
                
                # Pad the input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                
                # One-hot encode the output word
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                
                # Add to training data
                X1.append(feature)
                X2.append(in_seq)
                y.append(out_seq)
    
    # Convert to numpy arrays
    X1, X2, y = np.array(X1), np.array(X2), np.array(y)
    
    # Train the model
    model.fit([X1, X2], y, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model
def generate_caption(model, feature, tokenizer, max_length):
    # Seed the generation process with the start token
    in_text = 'startseq'
    
    # Iterate until end token or max length
    for i in range(max_length):
        # Encode the input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Predict the next word
        yhat = model.predict([feature, sequence], verbose=0)
        # Get the index with highest probability
        yhat = np.argmax(yhat)
        # Map the index to a word
        word = word_for_id(yhat, tokenizer)
        
        # Stop if we can't map the word
        if word is None:
            break
            
        # Add the word to the caption
        in_text += ' ' + word
        
        # Stop if we reach the end token
        if word == 'endseq':
            break
            
    # Remove the start and end tokens
    final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final_caption

def word_for_id(word_id, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == word_id:
            return word
    return None

# 2. Dataset Loading and Preparation
def load_dataset(dataset_path, images_path):
    # Dictionary to store image paths and captions
    image_captions = {}
    
    # Read the captions file
    with open(dataset_path, 'r') as f:
        for line in f:
            # Split line by comma
            parts = line.strip().split(',')
            image_id = parts[0]
            caption = parts[1]
            
            # Add start and end tokens
            caption = 'startseq ' + caption + ' endseq'
            
            # Add to dictionary
            if image_id not in image_captions:
                image_captions[image_id] = []
            image_captions[image_id].append(caption)
    
    # Return image captions dictionary
    return image_captions

# 3. Extract features for all images
def extract_all_features(feature_extractor, images_path, image_ids):
    features = {}
    for img_id in image_ids:
        img_path = f"{images_path}/{img_id}.jpg"
        features[img_id] = extract_features(feature_extractor, img_path)
    return features
def main():
    # 1. Configuration
    dataset_path = 'captions.txt'  # Path to caption dataset
    images_path = 'images/'        # Path to image folder
    max_length = 34                # Maximum caption length
    
    # 2. Load dataset
    print("Loading dataset...")
    captions = load_dataset(dataset_path, images_path)
    
    # 3. Create tokenizer
    print("Creating tokenizer...")
    tokenizer = create_tokenizer(captions)
    vocab_size = len(tokenizer.word_index) + 1
    
    # 4. Build feature extractor
    print("Building feature extractor...")
    feature_extractor = build_feature_extractor()
    
    # 5. Extract features for all images
    print("Extracting image features...")
    image_ids = list(captions.keys())
    features = extract_all_features(feature_extractor, images_path, image_ids)
    
    # 6. Build caption model
    print("Building caption model...")
    caption_model = build_caption_model(vocab_size, max_length)
    
    # 7. Train the model
    print("Training model...")
    trained_model = train_caption_model(
        caption_model, 
        features, 
        captions, 
        tokenizer,
        max_length,
        vocab_size,
        epochs=20,
        batch_size=64
    )
    
    # 8. Save the model
    print("Saving model...")
    trained_model.save('image_caption_model.h5')
    
    # 9. Test on a sample image
    test_image = 'test.jpg'
    print(f"Generating caption for {test_image}...")
    test_feature = extract_features(feature_extractor, test_image)
    test_caption = generate_caption(trained_model, test_feature, tokenizer, max_length)
    print(f"Generated caption: {test_caption}")

if __name__ == "__main__":
    main()
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, Embedding, Dropout, add
from tensorflow.keras.models import Model

# 1. Positional encoding
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
    
    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

# 2. Transformer Encoder Block
def encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-head attention
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    
    # Skip connection 1
    attention_output = add([inputs, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    
    # Feed Forward network
    ffn_output = Dense(ff_dim, activation="relu")(attention_output)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    
    # Skip connection 2
    ffn_output = add([attention_output, ffn_output])
    sequence_output = LayerNormalization(epsilon=1e-6)(ffn_output)
    
    return sequence_output

# 3. Transformer Decoder Block
def decoder_block(inputs, context, head_size, num_heads, ff_dim, dropout=0):
    # Self attention
    self_attention = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    
    # Skip connection 1
    self_attention = add([inputs, self_attention])
    self_attention = LayerNormalization(epsilon=1e-6)(self_attention)
    
    # Cross attention
    cross_attention = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(self_attention, context)
    
    # Skip connection 2
    cross_attention = add([self_attention, cross_attention])
    cross_attention = LayerNormalization(epsilon=1e-6)(cross_attention)
    
    # Feed Forward network
    ffn_output = Dense(ff_dim, activation="relu")(cross_attention)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    
    # Skip connection 3
    ffn_output = add([cross_attention, ffn_output])
    sequence_output = LayerNormalization(epsilon=1e-6)(ffn_output)
    
    return sequence_output

# 4. Complete Transformer-based Caption Model
def build_transformer_caption_model(vocab_size, max_length, embed_dim=256, num_heads=8, ff_dim=512):
    # Image feature input
    image_input = Input(shape=(2048,))
    image_features = Dense(embed_dim)(image_input)
    image_features = tf.expand_dims(image_features, 1)  # Add sequence dimension
    
    # Text input
    text_input = Input(shape=(max_length,))
    text_embedding = Embedding(vocab_size, embed_dim)(text_input)
    
    # Positional encoding
    pos_encoding = positional_encoding(max_length, embed_dim)
    text_embedding = text_embedding + pos_encoding[:, :max_length, :]
    
    # Encoder (for image features)
    encoder_output = encoder_block(image_features, embed_dim//num_heads, num_heads, ff_dim)
    
    # Decoder
    decoder_output = decoder_block(text_embedding, encoder_output, embed_dim//num_heads, num_heads, ff_dim)
    
    # Output layer
    output = Dense(vocab_size, activation="softmax")(decoder_output)
    
    # Create model
    model = Model(inputs=[image_input, text_input], outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    
    return model