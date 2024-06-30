from utils import *
from dem import *
# nltk.download('punkt')
import torch
import torch.optim as optim

def train_dem(model, tokenizer, tokenizer_name, D1, D2, D3, D4, train_dataloader, epochs, lr, k_values, verbose, save_dem, device):

    model.train()
    D1.train()
    D2.train()
    D3.train()
    D4.train()

    best_model_state = None
    best_D1_state = None
    best_D2_state = None

    cross_ent = nn.CrossEntropyLoss()
    cross_ent_recon = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    D2_optimizer = optim.Adam(D2.parameters(), lr=lr)
    D3_optimizer = optim.Adam(D3.parameters(), lr=lr)
    rest_optimizer = optim.Adam(list(model.parameters()) + list(D1.parameters()) + list(D4.parameters()), lr=lr)

    all_epochs_losses_d2_succ = []
    all_epochs_losses_d3_succ = []
    all_epochs_losses_recon = []
    all_epochs_losses_d1_succ = []
    all_epochs_losses_d2_fail = []
    all_epochs_losses_d3_fail = []
    all_epochs_losses_d4_succ = []
    all_epochs_losses_comb = []

    for epoch in tqdm(range(epochs), desc='Epochs'):

        epoch_d2_succ_loss = 0
        epoch_d3_succ_loss = 0

        epoch_d1_succ_loss = 0
        epoch_d2_fail_loss = 0
        epoch_d3_fail_loss = 0
        epoch_d4_succ_loss = 0
        epoch_recon_loss = 0
        epoch_comb_loss = 0

        for batch in tqdm(train_dataloader, desc='Batches'):

            # BLOOM TOKENIZER: shape for both = (batch size x longest sequence), padding = 3
            input_encoded = batch['input_ids'].to(device)
            input_labels = batch['labels'].to(device)
            input_attention_mask = batch['attention_mask'].to(device)
            input_BOW = batch['B'].to(device)
            
            # train D2 to make good predictions

            _, _, fs = model(input_encoded, attention_mask=input_attention_mask)
            ps = D2(fs)
            # LOGITS OF D3 OUTPUT AND TRUE LABELS ARE CLASS INDICES
            ce_loss_D2 = cross_ent(ps, input_labels)
            D2_optimizer.zero_grad()
            ce_loss_D2.backward()
            D2_optimizer.step()
            epoch_d2_succ_loss += ce_loss_D2.item()

            # train D3 to make good predictions

            _, fu, _ = model(input_encoded, attention_mask=input_attention_mask)
            ku = D3(fu)
            # LOGITS OF D3 OUTPUT AND TRUE LABELS ARE PROBS
            ce_loss_D3 = cross_ent(ku, input_BOW)
            D3_optimizer.zero_grad()
            ce_loss_D3.backward()
            D3_optimizer.step()
            epoch_d3_succ_loss += ce_loss_D3.item()

            # train dem, D1 and D4

            output_logits, fu, fs = model(input_encoded, attention_mask=input_attention_mask)

            # batch size x 2
            pu = D1(fu)
            ps = F.softmax(D2(fs), dim=1)
            # batch size x vocab size
            ku = F.softmax(D3(fu), dim=1)
            ks = D4(fs)

            # LOGITS OF MODEL OUTPUT AND TRUE LABELS ARE TOKEN INDICES
            loss_rec = cross_ent_recon(output_logits.view(-1, output_logits.size(-1)), input_encoded.view(-1))
            # LOGITS OF D1 OUTPUT AND LABELS ARE CLASS INDICES --> TRY TO GET CORRECT
            loss_D1 = cross_ent(pu, input_labels)
            # SOFTMAXED LOGITS OF D2 OUTPUT --> TRY TO GET UNIFORM
            loss_D2 = -entropy(ps, ps)

            # SOFTMAXED LOGITS OF D3 OUTPUT --> TRY TO GET UNIFORM
            loss_D3 = -entropy(ku, ku)
            # LOGITS OF D4 OUTPUT AND LABELS ARE PROBS
            loss_D4 = cross_ent(ks, input_BOW)
            loss = k_values[0] * loss_rec + k_values[1] * loss_D1 + k_values[2] * loss_D2 + k_values[3] * loss_D3 + k_values[4] * loss_D4
            rest_optimizer.zero_grad()
            loss.backward()
            rest_optimizer.step()
            epoch_comb_loss += loss.item()
            epoch_recon_loss += loss_rec.item()
            epoch_d1_succ_loss += loss_D1.item()
            epoch_d2_fail_loss += loss_D2.item()
            epoch_d3_fail_loss += loss_D3.item()
            epoch_d4_succ_loss += loss_D4.item()
    
        avg_d2_succ_loss = epoch_d2_succ_loss / len(train_dataloader)
        avg_d3_succ_loss = epoch_d3_succ_loss / len(train_dataloader)

        avg_recon_loss = epoch_recon_loss / len(train_dataloader)
        avg_d1_succ_loss = epoch_d1_succ_loss / len(train_dataloader)
        avg_d2_fail_loss = epoch_d2_fail_loss / len(train_dataloader)
        avg_d3_fail_loss = epoch_d3_fail_loss / len(train_dataloader)
        avg_d4_succ_loss = epoch_d4_succ_loss / len(train_dataloader)
        avg_combined_loss = epoch_comb_loss / len(train_dataloader)

        all_epochs_losses_d2_succ.append(avg_d2_succ_loss)
        all_epochs_losses_d3_succ.append(avg_d3_succ_loss)
        all_epochs_losses_recon.append(avg_recon_loss)
        all_epochs_losses_d1_succ.append(avg_d1_succ_loss)
        all_epochs_losses_d2_fail.append(avg_d2_fail_loss)
        all_epochs_losses_d3_fail.append(avg_d3_fail_loss)
        all_epochs_losses_d4_succ.append(avg_d4_succ_loss)
        all_epochs_losses_comb.append(avg_combined_loss)

        if verbose:
            print('')
            print('ADVERSARIAL LOSSES:')
            print(f"Epoch {epoch+1}: Avg D2 success loss per batch = {avg_d2_succ_loss:.4f}")
            print(f"Epoch {epoch+1}: Avg D3 success loss per batch = {avg_d3_succ_loss:.4f}")
            print('')
            print('COOPERATIVE LOSSES:')
            print(f"Epoch {epoch+1}: Avg recon loss per batch = {avg_recon_loss:.4f}")
            print(f"Epoch {epoch+1}: Avg D1 success loss per batch = {avg_d1_succ_loss:.4f}")
            print(f"Epoch {epoch+1}: Avg D2 fail loss per batch = {avg_d2_fail_loss:.4f}")
            print(f"Epoch {epoch+1}: Avg D3 fail loss per batch = {avg_d3_fail_loss:.4f}")
            print(f"Epoch {epoch+1}: Avg D4 success loss per batch = {avg_d4_succ_loss:.4f}")
            print(f"Epoch {epoch+1}: Avg combined loss per batch = {avg_combined_loss:.4f}")

    if save_dem:
        best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        best_D1_state = {key: value.cpu().clone() for key, value in D1.state_dict().items()}
        best_D2_state = {key: value.cpu().clone() for key, value in D2.state_dict().items()}
        # torch.save(best_model_state, f'./Models/dem_{tokenizer_name}_{lr}.pth')
        # torch.save(best_D1_state, f'./Models/D1_{tokenizer_name}_{lr}.pth')
        # torch.save(best_D2_state, f'./Models/D2_{tokenizer_name}_{lr}.pth')
        torch.save(best_model_state, f'./Models/dem.pth')
        torch.save(best_D1_state, f'./Models/D1.pth')
        torch.save(best_D2_state, f'./Models/D2.pth')