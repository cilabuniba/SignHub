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
        gesture_attention_mask = torch.ones((self.max_length, self.total_keypoints)).float()  # Dummy gesture attention mask

        return text, text_attention_mask, gestures, gesture_attention_mask, idx

    def get_original_gestures(self, idx):
        return self.gestures_data[idx]


# Funzione di Collate Personalizzata
def custom_collate_fn(batch):
    texts, text_attention_masks, gestures, _, indices = zip(*batch)

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

    return (
        torch.stack(texts),
        torch.stack(text_attention_masks),
        torch.stack(padded_gestures),
        torch.stack(gesture_attention_masks),
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
        for text, text_attention_mask, gestures, gesture_attention_mask, _ in val_loader:
            text = text.to(device)
            text_attention_mask = text_attention_mask.to(device)
            gestures = gestures.to(device)
            gesture_attention_mask = gesture_attention_mask.to(device)

            generated_gestures = model(text, text_attention_mask, gestures, gesture_attention_mask)

            loss = criterion(generated_gestures, gestures)
            running_loss += loss.item()

    return running_loss / len(val_loader)


# Esempio completo di training e inferenza
if __name__ == "__main__":
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Frasi di esempio
    phrases = ["How are you?",
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
               "What's going on?"]

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
    #train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

    # Definizione del dispositivo (GPU o CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Creazione del modello
    model = GestureGenerationModel(text_dim=768, hidden_dim=512, total_keypoints=48, num_layers=6).to(device)

    # Definizione dell'ottimizzatore
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Definizione della loss
    criterion = MPJPELoss()

    # Addestramento del modello
    num_epochs = 200
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for text, text_attention_mask, gestures, gesture_attention_mask, _ in train_loader:
            text = text.to(device)
            text_attention_mask = text_attention_mask.to(device)
            gestures = gestures.to(device)
            gesture_attention_mask = gesture_attention_mask.to(device)

            optimizer.zero_grad()

            generated_gestures = model(text, text_attention_mask, gestures, gesture_attention_mask)

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

        initial_gesture_attention_mask = torch.ones(max_length).unsqueeze(0).to(device)

        with torch.no_grad():
            generated_gestures = model(text_input, text_attention_mask, original_gestures, initial_gesture_attention_mask)

        return generated_gestures.squeeze(0).cpu().numpy()


    # Esempio di inferenza con il dataset
    text_inference = "how are you ?"
    generated_gestures = infer(text_inference, dataset)
    print("Generated Gestures for inference:")
    print(generated_gestures)


    def decode_without_special_tokens(text_tensor):
        tokens = tokenizer.convert_ids_to_tokens(text_tensor)
        filtered_tokens = [token for token in tokens if token not in ["[CLS]", "[SEP]","[PAD]"]]
        return tokenizer.convert_tokens_to_string(filtered_tokens)

    # Stampa i gesti originali relativi alla frase di inferenza
    for text, text_attention_mask, gestures, gesture_attention_mask, idx in train_loader:
        decoded_text = decode_without_special_tokens(text[0])
        print(decoded_text)
        if decoded_text == text_inference:
            original_gestures = dataset.get_original_gestures(idx[0]).numpy()
            print("Original Gestures:")
            print(original_gestures)
        break
