import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


class Trainer:
    """Neural Network Training Module with improved performance"""

    def __init__(self, config=None):
        # Default configuration with improved parameters
        self.config = {
            "hidden_size": 384,
            "learning_rate": 0.00406568155866974,  # Use this optimized value
            "batch_size": 1024,
            "weight_decay": 1.0895579988345462e-06,  # Use this optimized value
            "dropout_rate": 0.3880649861019966,  # Use this optimized value
            "output_dir": "models",
            "epochs": 50,
            "val_split": 0.1,
            "num_workers": 6,
            "save_freq": 1,
            "scheduler": "one_cycle",  # Keep the better scheduler
            "use_amp": True,
            "early_stopping": 10,
            "warmup_epochs": 3,
            "score_scale": 1000.0,
            "gradient_clip": 0.5,  # Keep gradient clipping
            "label_smoothing": 0.1,  # Keep label smoothing
            "optimizer": "adamw",  # Keep the better optimizer
        }
        # Update with provided config
        if config:
            self.config.update(config)

        # Set device with memory pinning for faster data transfer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            # Set for deterministic results if needed
            # torch.backends.cudnn.deterministic = True
        else:
            self.config['use_amp'] = False

        # Create output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(self.config['output_dir'], f"run_{self.timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        # Save configuration
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

        # Log file
        self.log_file = os.path.join(self.log_dir, "training_log.csv")
        with open(self.log_file, 'w') as f:
            f.write("epoch,train_loss,train_accuracy,train_speed,val_loss,val_accuracy,val_speed,learning_rate\n")

        self.scaler = GradScaler() if self.config['use_amp'] else None

        # Better loss function with label smoothing
        self.criterion = nn.MSELoss() if self.config['label_smoothing'] == 0 else self._smooth_mse_loss

        # Track best metrics
        self.best_val_loss = float('inf')
        self.best_accuracy = 0.0

    def _smooth_mse_loss(self, output, target):
        """MSE loss with label smoothing for better generalization"""
        # Apply label smoothing
        smooth_factor = self.config['label_smoothing']
        target = target * (1 - smooth_factor)
        return nn.MSELoss()(output, target)

    def _create_optimizer(self, model):
        """Create optimizer with improved settings"""
        if self.config['optimizer'] == 'adamw':
            # AdamW has better weight decay handling
            return optim.AdamW(
                model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                betas=(0.9, 0.999),  # Default betas, but explicitly set
                eps=1e-8
            )
        elif self.config['optimizer'] == 'sgd':
            # SGD with momentum and nesterov
            return optim.SGD(
                model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                momentum=0.9,
                nesterov=True
            )
        else:  # Default to Adam
            return optim.Adam(
                model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8
            )

    def _create_scheduler(self, optimizer, steps_per_epoch):
        """Create improved learning rate scheduler"""
        if self.config['scheduler'] == 'plateau':
            return ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True,
                min_lr=1e-6
            )
        elif self.config['scheduler'] == 'one_cycle':
            # One cycle policy often leads to faster convergence
            return OneCycleLR(
                optimizer,
                max_lr=self.config['learning_rate'],
                steps_per_epoch=steps_per_epoch,
                epochs=self.config['epochs'],
                pct_start=0.3,  # Spend 30% of time in warmup
                div_factor=25,  # initial_lr = max_lr/25
                final_div_factor=10000  # min_lr = initial_lr/10000
            )
        else:  # cosine
            return CosineAnnealingLR(
                optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config['learning_rate'] / 20
            )

    def _prepare_batch(self, batch):
        """Move batch data to device with non-blocking transfer"""
        return {
            'white_features': batch['white_features'].to(self.device, non_blocking=True),
            'black_features': batch['black_features'].to(self.device, non_blocking=True),
            'side_to_move': batch['side_to_move'].to(self.device, non_blocking=True),
            'score': self._prepare_target(batch['score'].to(self.device, non_blocking=True))
        }

    def _prepare_target(self, target):
        """Prepare target values with proper scaling"""
        return torch.clamp(target, -self.config['score_scale'], self.config['score_scale'])

    def train_epoch(self, model, train_loader, optimizer, scheduler=None):
        """Train for one epoch with improved performance"""
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_positions = 0
        start_time = time.time()

        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Training", unit="batch")

        for batch in progress_bar:
            # Move data to device efficiently
            processed_batch = self._prepare_batch(batch)
            batch_size = processed_batch['white_features'].size(0)

            # Forward pass with mixed precision
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            if self.config['use_amp']:
                with autocast():
                    output = model(
                        processed_batch['white_features'],
                        processed_batch['black_features'],
                        processed_batch['side_to_move']
                    )
                    output = output.view(-1)
                    loss = self.criterion(output, processed_batch['score'])

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Clip gradients to prevent explosion
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.config['gradient_clip'])

                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                output = model(
                    processed_batch['white_features'],
                    processed_batch['black_features'],
                    processed_batch['side_to_move']
                )
                output = output.view(-1)
                loss = self.criterion(output, processed_batch['score'])
                loss.backward()

                # Clip gradients
                nn.utils.clip_grad_norm_(model.parameters(), self.config['gradient_clip'])
                optimizer.step()

            # Update OneCycleLR if being used
            if scheduler is not None and isinstance(scheduler, OneCycleLR):
                scheduler.step()

            # Update statistics
            running_loss += loss.item() * batch_size
            total_positions += batch_size

            # Calculate accuracy (sign prediction)
            with torch.no_grad():
                pred_sign = torch.sign(output)
                target_sign = torch.sign(processed_batch['score'])
                correct_predictions += (pred_sign == target_sign).sum().item()

            # Update progress bar with more metrics
            positions_per_sec = total_positions / (time.time() - start_time + 1e-6)
            current_accuracy = correct_predictions / total_positions
            current_loss = running_loss / total_positions
            current_lr = optimizer.param_groups[0]['lr']

            progress_bar.set_postfix({
                'loss': f"{current_loss:.6f}",
                'acc': f"{current_accuracy:.2%}",
                'speed': f"{positions_per_sec:.1f} pos/s",
                'lr': f"{current_lr:.6f}"
            })

        # Calculate epoch statistics
        epoch_loss = running_loss / total_positions
        accuracy = correct_predictions / total_positions
        total_time = time.time() - start_time + 1e-6

        return {
            'loss': epoch_loss,
            'accuracy': accuracy,
            'total_positions': total_positions,
            'speed': total_positions / total_time
        }

    def validate(self, model, val_loader):
        """Validate with improved performance"""
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_positions = 0
        start_time = time.time()

        # For validation metrics
        all_outputs = []
        all_targets = []

        # Progress bar
        progress_bar = tqdm(val_loader, desc=f"Validating", unit="batch")

        with torch.no_grad():
            for batch in progress_bar:
                # Move data to device efficiently
                processed_batch = self._prepare_batch(batch)
                batch_size = processed_batch['white_features'].size(0)

                # Forward pass with mixed precision if enabled
                if self.config['use_amp']:
                    with autocast():
                        output = model(
                            processed_batch['white_features'],
                            processed_batch['black_features'],
                            processed_batch['side_to_move']
                        )
                        output = output.view(-1)
                        loss = self.criterion(output, processed_batch['score'])
                else:
                    output = model(
                        processed_batch['white_features'],
                        processed_batch['black_features'],
                        processed_batch['side_to_move']
                    )
                    output = output.view(-1)
                    loss = self.criterion(output, processed_batch['score'])

                # Store predictions and targets for analysis
                all_outputs.append(output.cpu())
                all_targets.append(processed_batch['score'].cpu())

                # Update statistics
                running_loss += loss.item() * batch_size
                total_positions += batch_size

                # Calculate accuracy (sign of evaluation matches)
                pred_sign = torch.sign(output)
                target_sign = torch.sign(processed_batch['score'])
                correct_predictions += (pred_sign == target_sign).sum().item()

                # Update progress bar
                current_loss = running_loss / total_positions
                current_accuracy = correct_predictions / total_positions

                progress_bar.set_postfix({
                    'loss': f"{current_loss:.6f}",
                    'acc': f"{current_accuracy:.2%}"
                })

        # Calculate validation statistics
        validation_time = time.time() - start_time
        val_loss = running_loss / total_positions
        accuracy = correct_predictions / total_positions

        # Return more comprehensive metrics
        return {
            'loss': val_loss,
            'accuracy': accuracy,
            'total_positions': total_positions,
            'time': validation_time,
            'speed': total_positions / validation_time if validation_time > 0 else 0
        }

    def train(self, model, train_loader, val_loader, test_loader=None):
        """Main training function with improved training routine"""
        print(f"Training on {self.device} with configuration:")
        for k, v in self.config.items():
            print(f"  {k}: {v}")

        # Move model to device
        model = model.to(self.device)

        # Create optimizer
        optimizer = self._create_optimizer(model)

        # Create scheduler based on training steps
        steps_per_epoch = len(train_loader)
        scheduler = self._create_scheduler(optimizer, steps_per_epoch)

        # Determine if scheduler should be stepped per batch or per epoch
        step_scheduler_per_batch = isinstance(scheduler, OneCycleLR)

        # Track metrics for early stopping
        early_stop_counter = 0
        start_time = time.time()

        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")

            # Train for one epoch with scheduler if step_per_batch
            train_results = self.train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler if step_scheduler_per_batch else None
            )

            # Validate
            val_results = self.validate(model, val_loader)

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Update learning rate scheduler (if not already updated per batch)
            if not step_scheduler_per_batch:
                if self.config['scheduler'] == 'plateau':
                    scheduler.step(val_results['loss'])
                else:
                    scheduler.step()

            # Print progress with more details
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1} Summary - "
                  f"Train Loss: {train_results['loss']:.6f}, "
                  f"Train Acc: {train_results['accuracy']:.2%}, "
                  f"Val Loss: {val_results['loss']:.6f}, "
                  f"Val Acc: {val_results['accuracy']:.2%}, "
                  f"LR: {current_lr:.6f}, "
                  f"Time: {epoch_time:.2f}s")

            # Log metrics
            with open(self.log_file, 'a') as f:
                f.write(f"{epoch + 1},"
                        f"{train_results['loss']},"
                        f"{train_results['accuracy']},"
                        f"{train_results['speed']},"
                        f"{val_results['loss']},"
                        f"{val_results['accuracy']},"
                        f"{val_results['speed']},"
                        f"{current_lr}\n")

            # Save best model - either by loss or accuracy depending on task
            improve_loss = val_results['loss'] < self.best_val_loss
            improve_acc = val_results['accuracy'] > self.best_accuracy

            if improve_loss:
                self.best_val_loss = val_results['loss']
                model_path = os.path.join(self.log_dir, f"nnue_best_loss.pt")
                model.save(model_path)
                print(f"✓ Saved best loss model (val_loss: {self.best_val_loss:.6f})")

            if improve_acc:
                self.best_accuracy = val_results['accuracy']
                model_path = os.path.join(self.log_dir, f"nnue_best_acc.pt")
                model.save(model_path)
                print(f"✓ Saved best accuracy model (val_acc: {self.best_accuracy:.2%})")

            # Early stopping logic
            if not (improve_loss or improve_acc):
                early_stop_counter += 1
                if early_stop_counter >= self.config['early_stopping']:
                    print(
                        f"! Early stopping after {epoch + 1} epochs (no improvement for {self.config['early_stopping']} epochs)")
                    break
            else:
                early_stop_counter = 0  # Reset early stopping counter

            # Save checkpoint every N epochs
            if (epoch + 1) % self.config['save_freq'] == 0 or epoch == self.config['epochs'] - 1:
                model_path = os.path.join(self.log_dir, f"nnue_epoch_{epoch + 1}.pt")
                model.save(model_path)
                print(f"✓ Saved checkpoint")

        # Final evaluation on test set if provided
        final_results = {}
        if test_loader:
            print("\nEvaluating on test set...")
            test_results = self.validate(model, test_loader)
            print(f"Test Loss: {test_results['loss']:.6f}, Test Accuracy: {test_results['accuracy']:.2%}")
            final_results['test_loss'] = test_results['loss']
            final_results['test_accuracy'] = test_results['accuracy']

        # Save final model
        final_model_path = os.path.join(self.log_dir, f"nnue_final.pt")
        model.save(final_model_path)
        print(f"✓ Saved final model")

        # Print training summary
        total_time = time.time() - start_time
        print(f"\nTraining Summary:")
        print(f"=====================================")
        print(f"Total training time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Best validation accuracy: {self.best_accuracy:.2%}")
        if test_loader:
            print(f"Final test loss: {test_results['loss']:.6f}")
            print(f"Final test accuracy: {test_results['accuracy']:.2%}")
        print(f"Models saved to: {self.log_dir}")

        # Save summary to file
        with open(os.path.join(self.log_dir, "summary.txt"), 'w') as f:
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total training time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)\n")
            f.write(f"Best validation loss: {self.best_val_loss:.6f}\n")
            f.write(f"Best validation accuracy: {self.best_accuracy:.2%}\n")
            if test_loader:
                f.write(f"Final test loss: {test_results['loss']:.6f}\n")
                f.write(f"Final test accuracy: {test_results['accuracy']:.2%}\n")
            f.write("\nConfiguration:\n")
            for k, v in self.config.items():
                f.write(f"  {k}: {v}\n")

        final_results.update({
            'best_val_loss': self.best_val_loss,
            'best_accuracy': self.best_accuracy,
            'model_path': final_model_path,
            'log_dir': self.log_dir
        })

        return final_results