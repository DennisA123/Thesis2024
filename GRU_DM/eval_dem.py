from tqdm import tqdm
import torch
from utils import *

def eval_dem_trained(model, test_dataloader, device, n):

    model.eval()

    total_correct_D1 = 0
    total_correct_D2 = 0
    total_samples = 0

    fu_samples = {0: [], 1: []}
    fs_samples = {0: [], 1: []}

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Batches'):

            input_encoded, input_labels = batch
            input_encoded = input_encoded.to(device)
            input_labels = input_labels.to(device)

            fu, fs = model.forward_encoder_preclass(input_encoded)

            # FOR VISUALIZING
            for label in [0, 1]:
                indices = (input_labels == label).nonzero(as_tuple=True)[0]
                fu_label = fu[indices]
                fs_label = fs[indices]

                if len(fu_samples[label]) < n:
                    fu_samples[label].extend(fu_label[:max(0, n - len(fu_samples[label]))].detach().cpu())

                if len(fs_samples[label]) < n:
                    fs_samples[label].extend(fs_label[:max(0, n - len(fs_samples[label]))].detach().cpu())

            pu, ps, _, _ = model.forward_encoder(input_encoded)

            predictions_D1 = torch.argmax(pu, dim=1)
            predictions_D2 = torch.argmax(ps, dim=1)

            total_correct_D1 += torch.eq(predictions_D1, input_labels).sum().item()
            total_correct_D2 += torch.eq(predictions_D2, input_labels).sum().item()
            total_samples += input_labels.size(0)

    fu_samples = {label: torch.stack(samples) for label, samples in fu_samples.items()}
    fs_samples = {label: torch.stack(samples) for label, samples in fs_samples.items()}
 
    fu_tsne, fu_labels1 = prepare_tsne_data(fu_samples, n)
    fs_tsne, fs_labels1 = prepare_tsne_data(fs_samples, n)

    plot_tsne(fu_tsne, fu_labels1, 't-SNE of gender vectors', name='fu')
    plot_tsne(fs_tsne, fs_labels1, 't-SNE of semantic vectors', name='fs')

    accuracy_D1 = total_correct_D1 / total_samples
    accuracy_D2 = total_correct_D2 / total_samples

    return accuracy_D1, accuracy_D2