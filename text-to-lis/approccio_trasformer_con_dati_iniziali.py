import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizer, BertModel
import random


# Definizione della Loss Function
class MPJPELoss(nn.Module):
    def __init__(self):
        super(MPJPELoss, self).__init__()

    def forward(self, predicted, target):
        return torch.mean(torch.norm(predicted - target, dim=-1))


# Definizione del Modello
class GestureGenerationModel(nn.Module):
    def __init__(self, text_dim, hidden_dim, total_keypoints, num_layers):
        super(GestureGenerationModel, self).__init__()
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_projector = nn.Linear(text_dim, hidden_dim)
        self.gesture_projector = nn.Linear(total_keypoints, hidden_dim)  # Aggiustato hidden_dim se necessario
        self.gesture_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=num_layers
        )
        self.output_projector = nn.Linear(hidden_dim, total_keypoints)
        self.total_keypoints = total_keypoints

    def forward(self, text, text_attention_mask, gesture_input, gesture_attention_mask):
        # Encode text
        text_output = self.text_encoder(text, attention_mask=text_attention_mask)
        text_embeds = text_output.last_hidden_state
        text_embeds = self.text_projector(text_embeds)

        # Project gestures
        gesture_input = self.gesture_projector(gesture_input)  # Assicurati che le dimensioni siano corrette

        # Adjust attention mask
        tgt_key_padding_mask = torch.logical_not(gesture_attention_mask).bool()

        # Pass through decoder
        output = self.gesture_decoder(
            gesture_input.permute(1, 0, 2),  # (L, N, E)
            text_embeds.permute(1, 0, 2),  # (S, N, E)
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,  # Corrected mask
            memory_key_padding_mask=(1 - text_attention_mask).bool()
        )

        # Project to output space
        generated_gestures = self.output_projector(output.permute(1, 0, 2))  # (N, L, E) -> (N, L, total_keypoints * 3)

        return generated_gestures


# Definizione del Dataset Sintetico
class SyntheticDataset(Dataset):
    def __init__(self, tokenizer, phrases, max_length=100, total_keypoints=48):
        self.tokenizer = tokenizer
        self.phrases = phrases
        self.max_length = max_length
        self.total_keypoints = total_keypoints

        self.gestures_data = []
        self.initial_gestures = []

        for _ in range(len(phrases)):
            initial_gestures = torch.zeros((max_length, total_keypoints))
            self.initial_gestures.append(initial_gestures)

            gestures = torch.rand((max_length, total_keypoints))
            self.gestures_data.append(gestures.clone())

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, idx):
        phrase = self.phrases[idx]
        inputs = self.tokenizer(phrase, return_tensors='pt', padding=True, truncation=True)
        text = inputs['input_ids'].squeeze(0)
        text_attention_mask = inputs['attention_mask'].squeeze(0)

        gestures = self.gestures_data[idx]
        gesture_attention_mask = torch.ones(
            (self.max_length, self.total_keypoints)).float()  # Dummy gesture attention mask

        initial_gesture = self.initial_gestures[idx]
        initial_gesture_mask = torch.ones((self.max_length, self.total_keypoints)).float()
        return text, text_attention_mask, gestures, gesture_attention_mask, initial_gesture, initial_gesture_mask, idx

    def get_original_gestures(self, idx):
        return self.initial_gestures[idx]


# Funzione di Collate Personalizzata
def custom_collate_fn(batch):
    texts, text_attention_masks, gestures, _, initial_gesture, _, indices = zip(*batch)

    if len(set(len(t) for t in texts)) > 1 or len(set(len(m) for m in text_attention_masks)) > 1:
        max_text_len = max(len(t) for t in texts)
        texts = [torch.cat([t, torch.zeros(max_text_len - len(t), dtype=torch.long)]) for t in texts]
        text_attention_masks = [torch.cat([m, torch.zeros(max_text_len - len(m), dtype=torch.long)]) for m in
                                text_attention_masks]

    max_gesture_len = max([len(g) for g in gestures])
    total_keypoints = gestures[0].shape[1] // 3
    padded_gestures = [pad_sequence(g, max_gesture_len) for g in gestures]

    # Generate key_padding_mask based on padded_gestures
    gesture_attention_masks = []
    for g in gestures:
        gesture_mask = torch.zeros_like(g)
        gesture_mask[g != 0] = 1  # Set to 1 where gestures are not padded
        gesture_attention_masks.append(gesture_mask.any(dim=-1))  # Create mask for packed sequence

    max_gesture_len = max([len(g) for g in initial_gesture])
    padded_initial_gestures = [pad_sequence(g, max_gesture_len) for g in initial_gesture]

    max_initial_gesture_len = max([len(g) for g in initial_gesture])
    initial_gesture_attention_masks = [torch.ones(max_initial_gesture_len) for _ in initial_gesture]

    return (
        torch.stack(texts),
        torch.stack(text_attention_masks),
        torch.stack(padded_gestures),
        torch.stack(gesture_attention_masks),
        torch.stack(padded_initial_gestures),
        torch.stack(initial_gesture_attention_masks),

        torch.tensor(indices)  # Return indices for accessing original gestures
    )


def pad_sequence(seq, max_len):
    padded_seq = torch.zeros((max_len, seq.shape[1]), dtype=torch.float32)
    padded_seq[:seq.shape[0], :] = seq
    return padded_seq


# Funzione di validazione
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for text, text_attention_mask, gestures, gesture_attention_mask, initial_gesture, initial_gesture_attention_mask, _ in val_loader:
            text = text.to(device)
            text_attention_mask = text_attention_mask.to(device)
            gestures = gestures.to(device)
            gesture_attention_mask = gesture_attention_mask.to(device)
            initial_gesture = initial_gesture.to(device)
            initial_gesture_attention_mask = initial_gesture_attention_mask.to(device)

            generated_gestures = model(text, text_attention_mask, initial_gesture, initial_gesture_attention_mask)

            loss = criterion(generated_gestures, gestures)
            running_loss += loss.item()

    return running_loss / len(val_loader)


# Esempio completo di training e inferenza
if __name__ == "__main__":
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Frasi di esempio
    phrases = [
        "How are you?",
        "Nice to meet you",
        "Can you help me?",
        "What's your name?",
        "Good morning",
        "See you later",
        "Thank you very much",
        "I don't understand",
        "Could you repeat that?",
        "Where is the nearest hospital?",
        "What time is it?",
        "I like this place",
        "How's the weather today?",
        "I'm sorry",
        "Excuse me",
        "Where are you from?",
        "Do you have a moment?",
        "Let's go for a walk",
        "I need a break",
        "Have a good day",
        "I'll call you later",
        "Please wait for me",
        "Let me think about it",
        "It's not a problem",
        "I appreciate your help",
        "Congratulations!",
        "I'm lost",
        "That's interesting",
        "Don't worry about it",
        "What do you think?",
        "I'll be right back",
        "Is everything okay?",
        "How do you feel?",
        "I have an idea",
        "That's a good point",
        "Let's get started",
        "I need some advice",
        "What's the plan?",
        "Can you hear me?",
        "What's going on?",
        "How was your day?",
        "Can I ask you something?",
        "Do you speak English?",
        "Please come in",
        "What do you need?",
        "Do you like it here?",
        "It's nice to see you",
        "I'm very tired",
        "Do you have any plans?",
        "How can I help?",
        "Could you pass the salt?",
        "How old are you?",
        "Do you want to join us?",
        "How long will it take?",
        "Could you show me?",
        "I really enjoyed that",
        "Do you live nearby?",
        "Can I borrow your phone?",
        "What are you doing?",
        "Can you drive?",
        "Do you need any help?",
        "I love this song",
        "What's your favorite color?",
        "Where do you work?",
        "Do you have any siblings?",
        "Can I sit here?",
        "Would you like some coffee?",
        "Do you know the way?",
        "What did you say?",
        "What is your favorite food?",
        "Where did you go?",
        "Can I help you with that?",
        "Are you feeling okay?",
        "When is your birthday?",
        "Do you like to read?",
        "Can I get you anything?",
        "What is your phone number?",
        "Can you swim?",
        "Do you play any instruments?",
        "Can you cook?",
        "Where do you live?",
        "What time do you wake up?",
        "Do you exercise?",
        "What's your favorite movie?",
        "How was your weekend?",
        "Do you have any pets?",
        "Where are we going?",
        "What do you like to do?",
        "Can you dance?",
        "Do you watch TV?",
        "How do you get to work?",
        "Do you like to travel?",
        "What did you eat for breakfast?",
        "Can you keep a secret?",
        "What are your hobbies?",
        "Where did you grow up?",
        "Can you speak any other languages?",
        "What's your favorite book?",
        "Do you like sports?",
        "What time is your meeting?",
        "Can you read this?",
        "What's your email address?",
        "Can you help me move?",
        "What do you think about this?",
        "Where do you want to go?",
        "Do you enjoy cooking?",
        "Can you pick me up?",
        "What's your favorite song?",
        "Do you like your job?",
        "How long have you lived here?",
        "Can you help me find this?",
        "What's your favorite hobby?",
        "Do you have a car?",
        "How did you meet?",
        "Can you write that down?",
        "What's your favorite drink?",
        "Do you like to paint?",
        "How far is it?",
        "Can you come with me?",
        "What's your favorite season?",
        "Do you like to hike?",
        "How often do you travel?",
        "Can you help me with my homework?",
        "What's your favorite holiday?",
        "Do you like to draw?",
        "How do you spend your weekends?",
        "Can you tell me more?",
        "What's your favorite animal?",
        "Do you like to sing?",
        "How was your trip?",
        "Can you believe this?",
        "What's your favorite dessert?",
        "Do you like to garden?",
        "How did you learn that?",
        "Can you show me the way?",
        "What's your favorite snack?",
        "Do you like to fish?",
        "How often do you go out?",
        "Can you help me with this project?",
        "What's your favorite fruit?",
        "Do you like to bike?",
        "How do you relax?",
        "Can you hear that?",
        "What's your favorite vegetable?",
        "Do you like to swim?",
        "How was your night?",
        "Can you see that?",
        "What's your favorite sport?",
        "Do you like to camp?",
        "How did you find this place?",
        "Can you explain that?",
        "What's your favorite thing to do?",
        "Do you like to shop?",
        "How was your lunch?",
        "Can you meet me here?",
        "What's your favorite flower?",
        "Do you like to read books?",
        "How do you like your coffee?",
        "Can you tell me the time?",
        "What's your favorite color?",
        "Do you like to watch movies?",
        "How was your morning?",
        "Can you lend me a hand?",
        "What's your favorite place?",
        "Do you like to play games?",
        "How do you feel about that?",
        "Can you come over?",
        "What's your favorite memory?",
        "Do you like to write?",
        "How did you do that?",
        "Can you keep a secret?",
        "What's your favorite smell?",
        "Do you like to run?",
        "How often do you read?",
        "Can you recommend a book?",
        "What's your favorite time of day?",
        "Do you like to bake?",
        "How do you stay fit?",
        "Can you fix this?",
        "What's your favorite kind of music?",
        "Do you like to sew?",
        "How did you know?",
        "Can you hear me?",
        "What's your favorite TV show?",
        "Do you like to draw?",
        "How was your dinner?",
        "Can you help me study?",
        "What's your favorite way to relax?",
        "Do you like to ski?",
        "How often do you cook?",
        "Can you see the difference?",
        "What's your favorite way to spend a day off?",
        "Do you like to meditate?",
        "How was your workout?",
        "Can you help me find my keys?",
        "What's your favorite type of exercise?",
        "Do you like to do puzzles?",
        "How do you stay organized?",
        "Can you tell me a story?",
        "What's your favorite subject?",
        "Do you like to knit?",
        "How do you spend your free time?",
        "Can you give me some advice?",
        "What's your favorite website?",
        "Do you like to play chess?",
        "How was your commute?",
        "Can you help me with this task?",
        "What's your favorite part of the day?",
        "Do you like to play cards?",
        "How do you manage stress?",
        "Can you solve this problem?",
        "What's your favorite app?",
        "Do you like to play sports?",
        "How often do you exercise?",
        "Can you describe that?",
        "What's your favorite quote?",
        "Do you like to go to the beach?",
        "How was your nap?",
        "Can you help me with this form?",
        "What's your favorite childhood memory?",
        "Do you like to go to the gym?",
        "How often do you visit your family?",
        "Can you recommend a movie?",
        "What's your favorite type of cuisine?",
        "Do you like to visit museums?",
        "How do you stay motivated?",
        "Can you help me with this recipe?",
        "What's your favorite workout?",
        "Do you like to visit parks?",
        "How often do you go shopping?",
        "Can you help me with this document?",
        "What's your favorite travel destination?",
        "Do you like to go hiking?",
        "How do you start your day?",
        "Can you send me that file?",
        "What's your favorite restaurant?",
        "Do you like to go to concerts?",
        "How was your meeting?"
    ]

    # Creazione del dataset e DataLoader
    dataset = SyntheticDataset(tokenizer, phrases)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% per l'addestramento
    val_size = dataset_size - train_size  # 20% per la validazione

    # Indici per i set di addestramento e validazione
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, dataset_size))

    # Creazione dei subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

    # Definizione del dispositivo (GPU o CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Creazione del modello
    model = GestureGenerationModel(text_dim=768, hidden_dim=512, total_keypoints=48, num_layers=20).to(device)

    # Definizione dell'ottimizzatore
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Definizione della loss
    criterion = MPJPELoss()

    # Addestramento del modello
    num_epochs = 1000
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for text, text_attention_mask, gestures, gesture_attention_mask, initial_gesture, initial_gesture_attention_mask, _ in train_loader:
            text = text.to(device)
            text_attention_mask = text_attention_mask.to(device)
            gestures = gestures.to(device)
            gesture_attention_mask = gesture_attention_mask.to(device)

            optimizer.zero_grad()
            initial_gesture = initial_gesture.to(device)
            initial_gesture_attention_mask = initial_gesture_attention_mask.to(device)

            generated_gestures = model(text, text_attention_mask, initial_gesture,initial_gesture_attention_mask)

            loss = criterion(generated_gestures, gestures)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation step
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "gesture_generation_model.pth")

    # Caricamento del modello migliore
    model.load_state_dict(torch.load("gesture_generation_model.pth"))


    # Inferenza su esempio di testo
    def infer(text_inference, dataset):
        model.eval()
        inputs = tokenizer(text_inference, return_tensors='pt', padding=True, truncation=True)
        text_input = inputs['input_ids'].to(device)
        text_attention_mask = inputs['attention_mask'].to(device)

        max_length = 100  # Lunghezza massima delle sequenze di gesti

        # Ottieni gli originali gesti associati alla frase di inferenza
        # for text, text_attention_mask, gestures, gesture_attention_mask, initial_gesture, initial_gesture_attention_mask, _ in train_loader:
        # decoded_text = decode_without_special_tokens(text[0])

        original_gestures = dataset.get_original_gestures(0).unsqueeze(0).to(device)

        initial_gesture_attention_masks = []
        for g in original_gestures:
            initial_gesture_mask = torch.zeros_like(g)
            initial_gesture_mask[g != 0] = 1  # Set to 1 where gestures are not padded
            initial_gesture_attention_masks.append(initial_gesture_mask.any(dim=-1))  # Create mask for packed sequence

        initial_gesture_attention_mask = torch.ones(max_length).unsqueeze(0).to(device)

        with torch.no_grad():
            generated_gestures = model(text_input, text_attention_mask, original_gestures,
                                       initial_gesture_attention_mask)

        return generated_gestures.squeeze(0).cpu().numpy()


    # Esempio di inferenza con il dataset
    text_inference = "how are you ?"
    generated_gestures = infer(text_inference, dataset)
    print("Generated Gestures for inference:")
    print(generated_gestures)


    def decode_without_special_tokens(text_tensor):
        tokens = tokenizer.convert_ids_to_tokens(text_tensor)
        filtered_tokens = [token for token in tokens if token not in ["[CLS]", "[SEP]", "[PAD]"]]
        return tokenizer.convert_tokens_to_string(filtered_tokens)


    # Stampa i gesti originali relativi alla frase di inferenza
    for text, text_attention_mask, gestures, gesture_attention_mask, initial_gesture, initial_gesture_attention_mask, _ in train_loader:
        decoded_text = decode_without_special_tokens(text[0])
        print(decoded_text)
        if decoded_text == text_inference:
            original_gestures = gestures
            print("Original Gestures:")
            print(original_gestures[0])  # Print only the first gesture
            break  # Exit the loop after printing the first matching gesture