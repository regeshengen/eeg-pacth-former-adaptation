%%writefile main_modified.py

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import sys 
from sklearn.metrics import roc_auc_score, f1_score 
import pickle 
from sklearn.model_selection import GroupShuffleSplit 
import matplotlib.pyplot as plt # Adicionado para plotagem

from dataset_tu_berlin import TUBerlinDataset # get_participant_splits não é usado no LOSO principal
from EEG_PatchFormer import PatchFormer 

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args, model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    if not train_loader or len(train_loader.dataset) == 0:
        # print(f"Epoch {epoch}: Training loader/dataset is empty. Skipping training.")
        return 0, 0 
        
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) 
        optimizer.zero_grad()
        output = model(data) 
        loss = criterion(output, target)
        loss.backward()
        
        if args.clip_grad_norm > 0: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)
        
        if args.log_interval > 0 and batch_idx > 0 and (batch_idx +1) % args.log_interval == 0 : 
            print(f'Train Epoch: {epoch} [{ (batch_idx + 1) * len(data)}/{len(train_loader.dataset)} ({100. * (batch_idx + 1) / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    if total_samples == 0: 
        # print(f"Epoch {epoch}: No samples processed in training.")
        return 0,0 
        
    avg_loss = total_loss / total_samples
    accuracy = 100. * correct / total_samples
    current_lr = optimizer.param_groups[0]['lr']
    print(f'\nTrain set Epoch {epoch}: Avg loss: {avg_loss:.4f}, Acc: {correct}/{total_samples} ({accuracy:.2f}%), LR: {current_lr:.6f}\n')
    return avg_loss, accuracy

def validate(args, model, data_loader, criterion, device, mode="Validation", epoch_num=None):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    all_targets_list = []
    all_pred_classes_list = []
    all_pred_probs_list = []

    epoch_log_prefix = f"Epoch {epoch_num} " if epoch_num is not None else ""

    if not data_loader or len(data_loader.dataset) == 0:
        # print(f"{epoch_log_prefix}{mode} loader/dataset is empty. Skipping {mode.lower()}.")
        return 0, 0, 0.0, 0.0 

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data) 
            
            loss = criterion(output, target) 
            total_loss += loss.item() * data.size(0) 
            
            pred_classes = output.argmax(dim=1)
            correct += pred_classes.eq(target).sum().item()
            
            pred_probs = torch.softmax(output, dim=1)
            
            total_samples += data.size(0)

            all_targets_list.extend(target.cpu().numpy())
            all_pred_classes_list.extend(pred_classes.cpu().numpy())
            all_pred_probs_list.extend(pred_probs.cpu().numpy())
            
    if total_samples == 0:
        # print(f"{epoch_log_prefix}{mode} set: No samples processed.")
        return 0, 0, 0.0, 0.0
            
    avg_loss = total_loss / total_samples
    accuracy = 100. * correct / total_samples

    all_targets_np = np.array(all_targets_list)
    all_pred_classes_np = np.array(all_pred_classes_list)
    all_pred_probs_np = np.array(all_pred_probs_list)

    f1_macro = 0.0
    if len(all_targets_np) > 0: 
        f1_macro = f1_score(all_targets_np, all_pred_classes_np, average='macro', zero_division=0) * 100.0

    auc_macro = 0.0
    num_unique_targets = len(np.unique(all_targets_np))
    
    if num_unique_targets > 1 and all_pred_probs_np.ndim == 2 and all_pred_probs_np.shape[1] == args.num_classes:
        try:
            auc_macro = roc_auc_score(all_targets_np, all_pred_probs_np, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"Warning: Could not compute AUC for {mode} set (epoch {epoch_num}): {e}")
    elif len(all_targets_np) > 0 : 
        # print(f"Warning: AUC not computed for {mode} set (epoch {epoch_num}). Unique targets: {num_unique_targets} (need >1). Probs shape: {all_pred_probs_np.shape} (expected 2D with {args.num_classes} cols).")
        pass # Silencioso para não poluir demais se for comum

    print(f'{epoch_log_prefix}{mode} set: Avg loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, F1-macro: {f1_macro:.2f}%, AUC: {auc_macro:.4f}\n')
    return avg_loss, accuracy, f1_macro, auc_macro


def main():
    parser = argparse.ArgumentParser(description='EEG-PatchFormer LOSO CV Training')
    parser.add_argument('--data_path', type=str, 
                        default='/content/drive/MyDrive/EEG_data/eeg_data.pkl', 
                        help='Path to the preprocessed .pkl dataset file')
    parser.add_argument('--window_size', type=int, default=800, help='Window size (e.g., 4s * 200Hz)')
    parser.add_argument('--step_size', type=int, default=400, help='Step size for training dataset generation')
    parser.add_argument('--sampling_rate', type=int, default=200, help='Sampling rate of EEG data (Hz)')
    parser.add_argument('--internal_val_ratio', type=float, default=0.20, help='Ratio of (N-1) subjects for internal validation within a LOSO fold')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs_per_fold', type=int, default=50, help='Number of epochs to train for each LOSO fold')
    parser.add_argument('--lr', type=float, default=5e-5, help='Initial learning rate') 
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for AdamW optimizer')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Max norm for gradient clipping (0.0 to disable)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=50, help='Log training status every N batches') 
    parser.add_argument('--save_path', type=str, default='./loso_cv_runs', help='Path to save models for each fold')
    parser.add_argument('--metric_to_optimize', type=str, default='AUC', choices=['AUC', 'F1-macro', 'Accuracy'], 
                        help='Metric to monitor for saving best model and for LR scheduler.')
    parser.add_argument('--use_scheduler', action='store_true', default=True, help='Use ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='Patience for LR scheduler (epochs)') 
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping (epochs)')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes (nback, dsr, wg)') 
    parser.add_argument('--num_T', type=int, default=16, help='Output channels of temporal_cnn (k)') 
    parser.add_argument('--patch_time', type=int, default=20, help='Patch width for Patcher.unfold (0.1s * 200Hz)')
    parser.add_argument('--patch_step', type=int, default=10, help='Patch stride for Patcher.unfold (0.05s * 200Hz)')
    parser.add_argument('--dim_head', type=int, default=32, help='Embedding dimension per attention head') 
    parser.add_argument('--depth', type=int, default=2, help='Number of Transformer encoder layers') 
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads') 
    parser.add_argument('--dropout_rate', type=float, default=0.6, help='Dropout rate in PatchFormer')
    parser.add_argument('--plot_fold_metrics', action='store_true', default=True, help='Plot validation metrics for each LOSO fold')


    args = parser.parse_args()

    print("--- Parsed Arguments (LOSO CV) ---")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("----------------------------------")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.save_path and not os.path.exists(args.save_path):
        try: os.makedirs(args.save_path); print(f"Created save directory: {args.save_path}")
        except OSError as e: print(f"Error creating directory {args.save_path}: {e}.")

    with open(args.data_path, 'rb') as f:
        all_data_raw_list_for_participants = pickle.load(f)
    all_participant_ids = sorted(list(set([item['participant'] for item in all_data_raw_list_for_participants])))
    num_total_participants = len(all_participant_ids)
    print(f"Total unique participants for LOSO CV: {num_total_participants}. IDs: {all_participant_ids}")

    if num_total_participants < 2:
        print("ERROR: LOSO CV requires at least 2 participants. Halting.")
        return

    overall_test_metrics = {'acc': [], 'f1': [], 'auc': []}
    label_map_used = {'nback': 0, 'dsr': 1, 'wg': 2} 

    for i in range(num_total_participants):
        test_subject_id_list = [all_participant_ids[i]]
        train_val_subject_ids = [pid for pid in all_participant_ids if pid != test_subject_id_list[0]]

        print(f"\n--- LOSO FOLD {i+1}/{num_total_participants}: Testing on {test_subject_id_list[0]} ---")

        train_p_fold = []
        val_p_fold = []
        if len(train_val_subject_ids) == 0:
             print("Warning: No subjects left for training/validation in this fold. Skipping."); 
             overall_test_metrics['acc'].append(0); overall_test_metrics['f1'].append(0); overall_test_metrics['auc'].append(0)
             continue
        elif len(train_val_subject_ids) < 2 or args.internal_val_ratio <=0 or args.internal_val_ratio >=1: 
            # print("  Using all (N-1) subjects for training, no internal validation set for this fold.")
            train_p_fold = train_val_subject_ids
        else:
            num_train_val_subjects = len(train_val_subject_ids)
            internal_val_count = max(1, int(args.internal_val_ratio * num_train_val_subjects))
            if internal_val_count >= num_train_val_subjects : internal_val_count = num_train_val_subjects -1
            
            if internal_val_count <=0 : 
                 train_p_fold = train_val_subject_ids
            else:
                dummy_X_internal = np.arange(num_train_val_subjects)
                gss_internal = GroupShuffleSplit(n_splits=1, test_size=internal_val_count, random_state=args.seed + i)
                train_indices_internal, val_indices_internal = next(gss_internal.split(dummy_X_internal, groups=dummy_X_internal))
                train_p_fold = [train_val_subject_ids[k] for k in train_indices_internal]
                val_p_fold = [train_val_subject_ids[k] for k in val_indices_internal]
        
        # print(f"  Fold Train participants: {train_p_fold} ({len(train_p_fold)} subjects)")
        # print(f"  Fold Validation participants: {val_p_fold} ({len(val_p_fold)} subjects)")

        # print("  Creating datasets for this fold...")
        train_dataset_fold = TUBerlinDataset(
            pkl_file_path=args.data_path, participants_to_use=train_p_fold, mode=f'train_fold{i+1}',
            window_size=args.window_size, step_size=args.step_size, 
            label_map=label_map_used, sampling_rate=args.sampling_rate
        )
        val_loader_fold = None
        if val_p_fold:
            val_dataset_fold = TUBerlinDataset(
                pkl_file_path=args.data_path, participants_to_use=val_p_fold, mode=f'val_fold{i+1}',
                window_size=args.window_size, step_size=args.window_size, 
                label_map=label_map_used, sampling_rate=args.sampling_rate
            )
            if len(val_dataset_fold) > 0:
                val_loader_fold = DataLoader(val_dataset_fold, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        if len(train_dataset_fold) == 0:
            print(f"ERROR: Fold {i+1} Training dataset is empty. Skipping."); 
            overall_test_metrics['acc'].append(0); overall_test_metrics['f1'].append(0); overall_test_metrics['auc'].append(0)
            continue
            
        train_loader_fold = DataLoader(train_dataset_fold, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

        # print("  Initializing model for this fold...")
        num_eeg_channels_fold = train_dataset_fold.get_num_channels()
        if num_eeg_channels_fold == 0 and len(train_dataset_fold)>0: num_eeg_channels_fold = train_dataset_fold.data[0].shape[0]
        if num_eeg_channels_fold == 0: 
            print(f"ERROR: Fold {i+1} could not determine EEG channels. Skipping."); 
            overall_test_metrics['acc'].append(0); overall_test_metrics['f1'].append(0); overall_test_metrics['auc'].append(0)
            continue
        
        model_input_size_fold = (1, num_eeg_channels_fold, args.window_size)
        idx_graph_fold = [num_eeg_channels_fold]

        model_fold = PatchFormer(
            num_classes=args.num_classes, input_size=model_input_size_fold, 
            sampling_rate=args.sampling_rate, num_T=args.num_T, 
            patch_time=args.patch_time, patch_step=args.patch_step,
            dim_head=args.dim_head, depth=args.depth, heads=args.heads,
            dropout_rate=args.dropout_rate, idx_graph=idx_graph_fold
        ).to(device)
        
        optimizer_fold = torch.optim.AdamW(model_fold.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion_fold = nn.CrossEntropyLoss()
        scheduler_fold = None
        if args.use_scheduler and val_loader_fold and len(val_loader_fold.dataset) > 0:
            scheduler_mode = 'max' if args.metric_to_optimize in ['AUC', 'F1-macro', 'Accuracy'] else 'min'
            scheduler_fold = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_fold, mode=scheduler_mode, factor=0.2, 
                                                                   patience=args.scheduler_patience, verbose=False, min_lr=1e-7)
        
        # print(f"  Starting training for fold {i+1} ({args.epochs_per_fold} epochs)...")
        fold_val_epochs_history = []
        fold_val_accs_history = []
        fold_val_f1s_history = []
        fold_val_aucs_history = []
        fold_train_losses_history = []
        fold_val_losses_history = []

        best_val_metric_fold = -1.0
        epochs_no_improve_fold = 0
        best_model_state_fold = None

        for epoch in range(1, args.epochs_per_fold + 1):
            train_loss_f, train_acc_f = train(args, model_fold, train_loader_fold, optimizer_fold, criterion_fold, device, epoch)
            fold_train_losses_history.append(train_loss_f)
            
            current_val_metric_for_saving_f = 0.0
            val_metric_to_pass_to_scheduler_f = 0.0

            if val_loader_fold and len(val_loader_fold.dataset) > 0:
                val_loss_f, val_acc_f, val_f1_f, val_auc_f = validate(args, model_fold, val_loader_fold, criterion_fold, device, 
                                                                      mode=f"Val_Fold{i+1}", epoch_num=epoch)
                fold_val_epochs_history.append(epoch)
                fold_val_accs_history.append(val_acc_f)
                fold_val_f1s_history.append(val_f1_f)
                fold_val_aucs_history.append(val_auc_f)
                fold_val_losses_history.append(val_loss_f)

                if args.metric_to_optimize == "AUC": current_val_metric_for_saving_f = val_auc_f
                elif args.metric_to_optimize == "F1-macro": current_val_metric_for_saving_f = val_f1_f
                else: current_val_metric_for_saving_f = val_acc_f
                
                val_metric_to_pass_to_scheduler_f = val_loss_f if scheduler_fold and scheduler_fold.mode == 'min' else current_val_metric_for_saving_f

                if current_val_metric_for_saving_f > best_val_metric_fold:
                    best_val_metric_fold = current_val_metric_for_saving_f
                    epochs_no_improve_fold = 0
                    best_model_state_fold = model_fold.state_dict()
                else:
                    epochs_no_improve_fold += 1
                
                if scheduler_fold: scheduler_fold.step(val_metric_to_pass_to_scheduler_f)
                
                if epochs_no_improve_fold >= args.early_stopping_patience:
                    print(f"    Fold {i+1} Early stopping at epoch {epoch}.")
                    break
            else: 
                best_model_state_fold = model_fold.state_dict()
        
        if args.plot_fold_metrics and fold_val_epochs_history:
            plt.figure(figsize=(12, 8))
            plt.suptitle(f"Fold {i+1} - Test Subject: {test_subject_id_list[0]} - Validation Metrics", fontsize=16)
            plt.subplot(2, 2, 1); plt.plot(fold_val_epochs_history, fold_val_accs_history, marker='o', label='Val Acc'); plt.title('Accuracy'); plt.grid(True); plt.legend()
            plt.subplot(2, 2, 2); plt.plot(fold_val_epochs_history, fold_val_f1s_history, marker='o', label='Val F1'); plt.title('F1-macro'); plt.grid(True); plt.legend()
            plt.subplot(2, 2, 3); plt.plot(fold_val_epochs_history, fold_val_aucs_history, marker='o', label='Val AUC'); plt.title('AUC'); plt.grid(True); plt.legend()
            plt.subplot(2, 2, 4); 
            if fold_train_losses_history: plt.plot(range(1, len(fold_train_losses_history) + 1), fold_train_losses_history, marker='.', linestyle='--', label='Train Loss')
            plt.plot(fold_val_epochs_history, fold_val_losses_history, marker='o', label='Val Loss'); plt.title('Loss'); plt.grid(True); plt.legend()
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            if args.save_path:
                plot_save_path = os.path.join(args.save_path, f"fold_{i+1}_val_metrics.png")
                try: plt.savefig(plot_save_path); # print(f"  Saved validation metrics plot for fold {i+1} to {plot_save_path}")
                except Exception as e: print(f"  Error saving plot for fold {i+1}: {e}")
            plt.show() 


        print(f"  Testing fold {i+1} on subject {test_subject_id_list[0]}...")
        test_dataset_fold = TUBerlinDataset(
            pkl_file_path=args.data_path, participants_to_use=test_subject_id_list, mode=f'test_fold{i+1}',
            window_size=args.window_size, step_size=args.window_size,
            label_map=label_map_used, sampling_rate=args.sampling_rate
        )
        if len(test_dataset_fold) > 0:
            test_loader_fold = DataLoader(test_dataset_fold, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            if best_model_state_fold:
                model_fold.load_state_dict(best_model_state_fold)
            else: 
                print(f"   Fold {i+1}: No best model state from training. Using initial/last model weights.")

            test_loss_f, test_acc_f, test_f1_f, test_auc_f = validate(args, model_fold, test_loader_fold, criterion_fold, device, 
                                                                      mode=f"Test_Fold{i+1}")
            overall_test_metrics['acc'].append(test_acc_f)
            overall_test_metrics['f1'].append(test_f1_f)
            overall_test_metrics['auc'].append(test_auc_f)
        else:
            print(f"  Fold {i+1} Test dataset for subject {test_subject_id_list[0]} is empty. Recording 0 for metrics.")
            overall_test_metrics['acc'].append(0); overall_test_metrics['f1'].append(0); overall_test_metrics['auc'].append(0)
            
        if best_model_state_fold and args.save_path:
            fold_model_save_path = os.path.join(args.save_path, f"model_fold{i+1}_test_subj{test_subject_id_list[0]}.pt")
            try: torch.save(best_model_state_fold, fold_model_save_path); # print(f"  Saved best model for fold {i+1} to {fold_model_save_path}")
            except Exception as e: print(f"  Error saving model for fold {i+1}: {e}")


    print("\n--- LOSO Cross-Validation Finished ---")
    avg_test_acc = np.mean(overall_test_metrics['acc']) if overall_test_metrics['acc'] else 0
    std_test_acc = np.std(overall_test_metrics['acc']) if overall_test_metrics['acc'] else 0
    avg_test_f1 = np.mean(overall_test_metrics['f1']) if overall_test_metrics['f1'] else 0
    std_test_f1 = np.std(overall_test_metrics['f1']) if overall_test_metrics['f1'] else 0
    avg_test_auc = np.mean(overall_test_metrics['auc']) if overall_test_metrics['auc'] else 0
    std_test_auc = np.std(overall_test_metrics['auc']) if overall_test_metrics['auc'] else 0

    print(f"Average Test Accuracy across {num_total_participants} folds: {avg_test_acc:.2f}% (+/- {std_test_acc:.2f}%)")
    print(f"Average Test F1-macro across {num_total_participants} folds: {avg_test_f1:.2f}% (+/- {std_test_f1:.2f}%)")
    print(f"Average Test AUC across {num_total_participants} folds: {avg_test_auc:.4f} (+/- {std_test_auc:.4f})")

    print("\nIndividual fold test metrics:")
    for i in range(num_total_participants):
        if i < len(overall_test_metrics['acc']): # Check if metrics were recorded for this fold
            print(f"  Fold {i+1} (Test Subj: {all_participant_ids[i]}): Acc={overall_test_metrics['acc'][i]:.2f}%, F1={overall_test_metrics['f1'][i]:.2f}%, AUC={overall_test_metrics['auc'][i]:.4f}")
        else:
            print(f"  Fold {i+1} (Test Subj: {all_participant_ids[i]}): Metrics not available (fold possibly skipped).")


    if overall_test_metrics['acc']: 
        folds_ran = range(1, len(overall_test_metrics['acc']) + 1) # Use actual number of folds for which metrics were recorded
        plt.figure(figsize=(12, 7))
        
        plt.plot(folds_ran, overall_test_metrics['acc'], marker='o', linestyle='-', label='Test Accuracy per Fold')
        plt.plot(folds_ran, overall_test_metrics['f1'], marker='s', linestyle='--', label='Test F1-macro per Fold')
        auc_for_plot = [x * 100 for x in overall_test_metrics['auc']]
        plt.plot(folds_ran, auc_for_plot, marker='^', linestyle=':', label='Test AUC (*100) per Fold')
        
        plt.title(f'Final Test Metrics per LOSO Fold (Total Folds Run: {len(folds_ran)})')
        plt.xlabel('Fold Number')
        plt.ylabel('Metric Value (%)')
        # Create x-tick labels based on actual subjects tested in each fold
        x_tick_labels = [all_participant_ids[j] for j in range(len(folds_ran))] # Assumes folds run sequentially
        plt.xticks(folds_ran, x_tick_labels, rotation=45, ha="right")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 105) 
        plt.tight_layout()
        if args.save_path:
            summary_plot_path = os.path.join(args.save_path, "loso_summary_test_metrics.png")
            try: plt.savefig(summary_plot_path); print(f"Saved summary test metrics plot to {summary_plot_path}")
            except Exception as e: print(f"Error saving summary plot: {e}")
        plt.show()

if __name__ == '__main__':
    main()