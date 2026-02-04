import pandas as pd
import numpy as np
import random as r
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold, KFold 
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import f1_score
from tqdm import tqdm 
import os
import gc

'''
This code was optimized in a local environment with:
- Numpy 1.26.4
- Scikit-learn 1.8.0
- Torch 2.9.1

Kaggle Competition score: 0.47644 
'''

# Settings
def set_seed(seed=1):
    r.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(1) 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}") #this is added to use kaggle's gpu
#but cpu and gpu calculations make some difference on results
#for result consistency I generally used cpu on VSCode environment.

# Mixup - Data Augmentation: Applies by taking a linear combination of two samples
#Args -> x(tensor): input batch features y(tensor): input batch targets aplha: param for beta distr. to sample lambda 
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam 
        #mixed_x (Tensor): Mixed input features
        #y_a (Tensor): Original targets
        #y_b (Tensor): Targets of the shuffled batch
        #lam (float): The mixing coefficient

#Calculates loss for mixed inputs
#Args-> criterion: loss function (for example CrossEntropyLoss), pred (Tensor): Model predictions.
       # y_a (tensor): Original targets y_b (Tensor): Targets of the shuffled batch.
        #lam (float): The mixing coefficient.
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# MODEL: Attentive DAE-ResNet: Applies a channel-wise attention mechanism to the input features.
class FeatureAttention(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(FeatureAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        weight = self.fc(x)
        return x * weight


    """Denoising Autoencoder with ResNet connections and feature attention. Args like tihs:
        input_dim (int): Input feature dimension.
        num_classes (int): Number of output classes.
        hidden_dim (int): Dimension of the hidden layers.
        bottle_neck_dim (int): Dimension of the bottleneck layer.
        dropout_rate (float): Dropout probability."""  
class AttentiveCompactDAEResNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512, bottle_neck_dim=128, dropout_rate=0.3):
        super(AttentiveCompactDAEResNet, self).__init__()
        self.attention = FeatureAttention(input_dim)
        self.bn_input = nn.BatchNorm1d(input_dim)

        # encoder compresses input to bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, bottle_neck_dim),
            nn.BatchNorm1d(bottle_neck_dim),
            nn.LeakyReLU()
        )

        # decoder reconstructs input from bottleneck (Auxiliary task)
        self.decoder = nn.Sequential(
            nn.Linear(bottle_neck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Classification Head: predicts class labels from bottleneck
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(bottle_neck_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x_att = self.attention(x)
        x_norm = self.bn_input(x_att)
        latent = self.encoder(x_norm)
        reconstruction = self.decoder(latent)
        logits = self.head(latent)
        return logits, reconstruction

    """
    Trains the model for one epoch using Mixup and joint loss (Classification + Reconstruction).
        model (nn.Module): The neural network model.
        loader (DataLoader): Training data loader.
        optimizer: Optimizer instance.
        criterion_cls: Classification loss function.
        criterion_rec: Reconstruction loss function.
        alpha (float): Mixup alpha parameter.   
    Returns:
        float: Average loss for the epoch.
    """
def train_one_epoch(model, loader, optimizer, criterion_cls, criterion_rec, alpha):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        mixed_x, y_a, y_b, lam = mixup_data(X_batch, y_batch, alpha=alpha)
        optimizer.zero_grad()
        logits, reconstruction = model(mixed_x)
        loss_cls = mixup_criterion(criterion_cls, logits, y_a, y_b, lam)
        loss_rec = criterion_rec(reconstruction, mixed_x) 

        # Combine losses (0.1 weight for reconstruction is a hyperparameter)
        loss = loss_cls + 0.1 * loss_rec
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

#Evaluates model on the validation set
#Args->model (nn.Module): The neural network model. loader (DataLoader): Validation data loader.
def validate(model, loader):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            logits, _ = model(X_batch)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            targets.extend(y_batch.numpy())
    return f1_score(targets, preds, average='macro')

#Generates class probabilities for the test set and returns prob. matrix
def get_probs(model, X_test):
    model.eval()
    test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    dataset = TensorDataset(test_tensor)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(batch[0])
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs)

# MAIN PIPELINE
def main():
    print("Loading Data...")
    try:
        train_df = pd.read_csv("train.csv", header=0, index_col=0)
        test_df = pd.read_csv("test.csv", header=0, index_col=0)
    except FileNotFoundError:
        print("Error! Files not found")
        return

    X = train_df.iloc[:, :-2].values
    y = train_df.iloc[:, -2].values.astype(int)
    groups = train_df.iloc[:, -1].values
    X_test_kaggle_raw = test_df.values 
    
    print("Preprocessing...")
    scaler = QuantileTransformer(output_distribution='normal', random_state=42) #Using Quantile Transformer to normalize data to normal distribution
    combined = np.vstack((X, X_test_kaggle_raw))
    scaler.fit(combined)
    X = scaler.transform(X)
    X_test_kaggle = scaler.transform(X_test_kaggle_raw)

    INPUT_DIM = X.shape[1]
    NUM_CLASSES = 4 
    BATCH_SIZE = 128 
    EPOCHS = 50       
    LR = 1e-3
    MIXUP_ALPHA = 0.4
    
    # =========================================================
    # PHASE 0: STANDARD 5-FOLD CV
    # =========================================================
    print("\n--- STAGE 0: Standard 5-Fold Cross Validation ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=1) 
    std_f1_scores = [] 

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)): 
        X_tr, X_va = X[train_idx], X[val_idx] 
        y_tr, y_va = y[train_idx], y[val_idx] 
        
        tr_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)), batch_size=BATCH_SIZE, shuffle=True) 
        va_loader = DataLoader(TensorDataset(torch.FloatTensor(X_va), torch.LongTensor(y_va)), batch_size=BATCH_SIZE, shuffle=False) 
        
        cv_model = AttentiveCompactDAEResNet(INPUT_DIM, NUM_CLASSES).to(DEVICE) 
        cv_optimizer = torch.optim.AdamW(cv_model.parameters(), lr=LR, weight_decay=1e-3) 
        
        best_fold_f1 = 0 
        for epoch in range(20): # Used 29 epoch for faster CV results 
            train_one_epoch(cv_model, tr_loader, cv_optimizer, nn.CrossEntropyLoss(), nn.MSELoss(), alpha=MIXUP_ALPHA) 
            current_f1 = validate(cv_model, va_loader)
            if current_f1 > best_fold_f1: best_fold_f1 = current_f1 
            
        std_f1_scores.append(best_fold_f1) 
        print(f"Standard Fold {fold+1} F1: {best_fold_f1:.4f}") 

    # =========================================================
    # PHASE 1: TEACHER (Original Data -> Teacher)
    # =========================================================
    print("\n--- PHASE 1: Training Teacher (Attentive DAE-ResNet) ---")
    
    full_train_ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    full_train_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    teacher_model = AttentiveCompactDAEResNet(INPUT_DIM, NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_rec = nn.MSELoss()
    
    for epoch in tqdm(range(EPOCHS), desc="Training Teacher"):
        train_one_epoch(teacher_model, full_train_loader, optimizer, criterion_cls, criterion_rec, alpha=MIXUP_ALPHA)
        scheduler.step()
        
    teacher_probs = get_probs(teacher_model, X_test_kaggle)
    
    confidence_threshold = 0.90
    max_probs = np.max(teacher_probs, axis=1)
    pseudo_labels = np.argmax(teacher_probs, axis=1)
    high_conf_indices = np.where(max_probs >= confidence_threshold)[0]
    
    X_pseudo_1 = X_test_kaggle[high_conf_indices]
    y_pseudo_1 = pseudo_labels[high_conf_indices]
    
    print(f"\n-> Gen 1 Pseudo-Labeling Report:")
    print(f"   Samples Added for Student 1: {len(X_pseudo_1)}")
    
    # =========================================================
    # PHASE 2: STUDENT 1 (Teacher Labels -> Student 1)
    # =========================================================
    print("\n--- PHASE 2: Training Student Generation 1 (Group 5-Fold CV) ---") 
    
    test_probs_sum_gen1 = np.zeros((len(X_test_kaggle), NUM_CLASSES))
    gkf = GroupKFold(n_splits=5)
    f1_scores_gen1 = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train_orig, X_val = X[train_idx], X[val_idx]
        y_train_orig, y_val = y[train_idx], y[val_idx]
        
        X_train_final = np.vstack((X_train_orig, X_pseudo_1))
        y_train_final = np.concatenate((y_train_orig, y_pseudo_1))
        
        train_ds = TensorDataset(torch.FloatTensor(X_train_final), torch.LongTensor(y_train_final))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = AttentiveCompactDAEResNet(INPUT_DIM, NUM_CLASSES).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
        
        criterion_cls = nn.CrossEntropyLoss()
        criterion_rec = nn.MSELoss()
        
        best_f1 = 0
        best_state = None
        
        for epoch in range(EPOCHS): 
            loss = train_one_epoch(model, train_loader, optimizer, criterion_cls, criterion_rec, alpha=MIXUP_ALPHA)
            val_f1 = validate(model, val_loader)
            scheduler.step(val_f1)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = model.state_dict()
        
        print(f"Gen 1 - Group Fold {fold+1} Best F1: {best_f1:.4f}")
        f1_scores_gen1.append(best_f1)
        
        model.load_state_dict(best_state)
        test_probs_sum_gen1 += get_probs(model, X_test_kaggle)
        
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()

    # =========================================================
    # PHASE 3: STUDENT 2 (Student 1 Labels -> Student 2)
    # =========================================================
    print(f"\n--- PREPARING GENERATION 2: Boosting Student 1 Predictions ---")
    
    avg_probs_gen1 = test_probs_sum_gen1 / 5
    boosted_probs_gen1 = avg_probs_gen1.copy()
    boosted_probs_gen1[:, 3] *= 1.50
    pseudo_labels_gen2 = np.argmax(boosted_probs_gen1, axis=1)
    max_probs_gen2 = np.max(boosted_probs_gen1, axis=1)
    high_conf_indices_2 = np.where(max_probs_gen2 >= confidence_threshold)[0]
    
    X_pseudo_2 = X_test_kaggle[high_conf_indices_2]
    y_pseudo_2 = pseudo_labels_gen2[high_conf_indices_2]
    
    print(f"\n-> Gen 2 Pseudo-Labeling Report:")
    print(f"   Samples Added for Student 2: {len(X_pseudo_2)}")
    
    print(f"\n--- PHASE 3: Training Student 2 (Gen 2) ---")
    
    test_probs_sum_gen2 = np.zeros((len(X_test_kaggle), NUM_CLASSES))
    f1_scores_gen2 = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        tqdm.write(f"\nGen 2 - Fold {fold+1}...")
        X_train_orig, X_val = X[train_idx], X[val_idx]
        y_train_orig, y_val = y[train_idx], y[val_idx]
        X_train_final = np.vstack((X_train_orig, X_pseudo_2))
        y_train_final = np.concatenate((y_train_orig, y_pseudo_2))
        
        train_ds = TensorDataset(torch.FloatTensor(X_train_final), torch.LongTensor(y_train_final))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = AttentiveCompactDAEResNet(INPUT_DIM, NUM_CLASSES).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
        
        criterion_cls = nn.CrossEntropyLoss()
        criterion_rec = nn.MSELoss()
        
        best_f1 = 0
        best_state = None
        
        for epoch in tqdm(range(EPOCHS), desc=f"Fold {fold+1}", leave=False):
            loss = train_one_epoch(model, train_loader, optimizer, criterion_cls, criterion_rec, alpha=MIXUP_ALPHA)
            val_f1 = validate(model, val_loader)
            scheduler.step(val_f1)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = model.state_dict()
        
        tqdm.write(f"Gen 2 - Group Fold {fold+1} Best F1: {best_f1:.4f}")
        f1_scores_gen2.append(best_f1)
        model.load_state_dict(best_state)
        test_probs_sum_gen2 += get_probs(model, X_test_kaggle)
        
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()

    # =========================================================
    # FINAL RESULTS 
    # =========================================================
    print(f"\n==================================")
    print(f"Average Standard 5-Fold F1: {np.mean(std_f1_scores):.4f}") 
    print(f"Average Group 5-Fold F1: {np.mean(f1_scores_gen2):.4f}") 
    print(f"==================================")
    
    avg_probs_final = test_probs_sum_gen2 / 5
    final_probs = avg_probs_final.copy()
    final_probs[:, 3] *= 1.50
    final_preds = np.argmax(final_probs, axis=1)
    
    sub = pd.DataFrame({'ID': range(len(final_preds)), 'Predicted': final_preds})
    filename = "submission_final.csv" 
    sub.to_csv(filename, index=False)
    
    print(f"Final Submission Saved to: {filename}")

if __name__ == "__main__":
    main()
