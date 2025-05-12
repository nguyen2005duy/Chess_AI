import os
import sys
import torch
import random
import numpy as np
import time
from torch.utils.data import ConcatDataset, random_split, Subset
from dataloader import NNUEDataset
import torch.nn as nn
from torch.utils.data import DataLoader
from model import NNUE
from trainer2 import Trainer


def detect_training_phase(output_dir, total_phases=5):
    """
    Detect which phase to resume training from based on existing model checkpoints.

    Args:
        output_dir: Directory containing model phase folders
        total_phases: Total number of phases in the training pipeline

    Returns:
        tuple: (start_phase, model_path) where:
            - start_phase is the phase to resume training from (1 for full training)
            - model_path is the path to the best model from the previous phase (None if starting from scratch)
    """
    import os
    import glob

    print("\nChecking for existing model checkpoints...")

    # Default values (start from scratch)
    start_phase = 1
    model_path = None

    # Check each phase in reverse order to find the most advanced checkpoint
    for phase in range(total_phases, 0, -1):
        phase_dir = os.path.join(output_dir, f"phase_{phase}")

        if not os.path.exists(phase_dir):
            continue

        # Look for run directories within this phase
        run_dirs = sorted(glob.glob(os.path.join(phase_dir, "run_*")))

        if not run_dirs:
            continue

        # Get the most recent run
        latest_run = run_dirs[-1]

        # Check for model files in this run (prioritize best accuracy model)
        model_candidates = [
            os.path.join(latest_run, "nnue_best_acc.pt"),  # Best accuracy model
            os.path.join(latest_run, "nnue_best.pt"),  # Best loss model
            os.path.join(latest_run, "nnue_final.pt")  # Final model
        ]

        # Find the first existing model file
        for candidate in model_candidates:
            if os.path.exists(candidate):
                # Found a model! We should start from the next phase
                start_phase = phase + 1
                model_path = candidate

                print(f"Found checkpoint from phase {phase}: {os.path.basename(candidate)}")
                print(f"→ {candidate}")

                # Check if this was the final phase
                if start_phase > total_phases:
                    print(f"Training already completed all {total_phases} phases!")
                    print(f"To retrain the final phase, manually specify phase {total_phases}.")
                    start_phase = total_phases  # Set to final phase for retraining

                return start_phase, model_path

    # If we get here, no checkpoints were found
    print("No existing checkpoints found. Starting training from scratch (Phase 1).")
    return start_phase, model_path


def save_training_status(output_dir, phase, status, model_path=None, metrics=None):
    """
    Save training status information to a status file.

    Args:
        output_dir: Base output directory
        phase: Current training phase
        status: Status string ('in_progress', 'completed', 'failed')
        model_path: Path to the current best model (optional)
        metrics: Dictionary of metrics to save (optional)
    """
    import os
    import json
    import time

    status_file = os.path.join(output_dir, "training_status.json")

    # Load existing status if available
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status_data = json.load(f)
    else:
        status_data = {
            "training_history": [],
            "current_phase": None,
            "last_update": None
        }

    # Update with new status
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Add new entry to history
    history_entry = {
        "timestamp": timestamp,
        "phase": phase,
        "status": status,
    }

    if model_path:
        history_entry["model_path"] = model_path

    if metrics:
        history_entry["metrics"] = metrics

    status_data["training_history"].append(history_entry)

    # Update current phase and timestamp
    status_data["current_phase"] = phase
    status_data["last_update"] = timestamp
    status_data["last_status"] = status

    if model_path:
        status_data["last_model_path"] = model_path

    # Write updated status
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)

    print(f"Training status updated: Phase {phase} - {status}")


def get_training_status(output_dir):
    """
    Get current training status information.

    Args:
        output_dir: Base output directory

    Returns:
        dict: Training status information or None if no status file
    """
    import os
    import json

    status_file = os.path.join(output_dir, "training_status.json")

    if not os.path.exists(status_file):
        return None

    with open(status_file, 'r') as f:
        return json.load(f)


def print_training_status(output_dir):
    """
    Print a summary of the current training status.

    Args:
        output_dir: Base output directory
    """
    status_data = get_training_status(output_dir)

    if not status_data:
        print("No training status information available.")
        return

    print("\nTraining Status Summary:")
    print("=" * 60)
    print(f"Current Phase: {status_data.get('current_phase', 'None')}")
    print(f"Last Update: {status_data.get('last_update', 'None')}")
    print(f"Status: {status_data.get('last_status', 'Unknown')}")

    if 'last_model_path' in status_data:
        print(f"Latest Model: {status_data['last_model_path']}")

    print("\nTraining History:")
    for entry in status_data.get('training_history', [])[-5:]:  # Show last 5 entries
        print(f"- {entry['timestamp']}: Phase {entry['phase']} - {entry['status']}")

    print(f"\nTotal History Entries: {len(status_data.get('training_history', []))}")

    # Calculate total training time if possible
    history = status_data.get('training_history', [])
    if len(history) >= 2:
        import datetime
        try:
            start_time = datetime.datetime.strptime(history[0]['timestamp'], "%Y-%m-%d %H:%M:%S")
            end_time = datetime.datetime.strptime(history[-1]['timestamp'], "%Y-%m-%d %H:%M:%S")
            duration = end_time - start_time
            hours = duration.total_seconds() / 3600
            print(f"Total Training Time: {hours:.2f} hours")
        except:
            pass


def setup_datasets(data_dir, phase, curriculum_config=None):
    """
    Load and prepare datasets based on the training phase with curriculum learning

    Args:
        data_dir: Base directory containing the data folders
        phase: Current training phase (1-5)
        curriculum_config: Optional curriculum configuration for specialized training
    """
    # Modified ELO range distribution for a 5-phase approach with better diversity
    phase_distributions = {
        # Phase 1: Foundation - focus on mid-range ELO data for stable base training
        1: {
            "2000-2199": 0.70,  # 70% of data from 2000-2199 ELO (abundant data)
            "2200-2399": 0.25,  # 25% of data from 2200-2399 ELO
            "2400-2599": 0.05,  # 5% of data from 2400-2599 ELO (limited data)
        },
        # Phase 2: General Patterns - balanced use of mid/high ELO
        2: {
            "2000-2199": 0.55,  # 55% of data from 2000-2199 ELO
            "2200-2399": 0.35,  # 35% of data from 2200-2399 ELO
            "2400-2599": 0.10,  # 10% of data from 2400-2599 ELO
        },
        # Phase 3: Advanced Patterns - more balanced distribution
        3: {
            "2000-2199": 0.40,  # 40% of data from 2000-2199 ELO
            "2200-2399": 0.40,  # 40% of data from 2200-2399 ELO
            "2400-2599": 0.20,  # 20% of data from 2400-2599 ELO
        },
        # Phase 4: Expert Knowledge - more balanced, less extreme shift
        4: {
            "2000-2199": 0.30,  # Increased from 0.25 to 0.30
            "2200-2399": 0.45,  # 45% of data from 2200-2399 ELO
            "2400-2599": 0.25,  # Decreased from 0.30 to 0.25
        },
        # Phase 5: Fine-tuning - more diverse high-quality data
        5: {
            "2000-2199": 0.25,  # Increased from 0.15 to 0.25
            "2200-2399": 0.45,  # Increased from 0.40 to 0.45
            "2400-2599": 0.30,  # Decreased from 0.45 to 0.30
        }
    }

    # Apply curriculum configuration if provided (for specialized phases)
    if curriculum_config:
        if curriculum_config.get("distribution_override"):
            distribution = curriculum_config["distribution_override"]
        else:
            distribution = phase_distributions[phase]
    else:
        distribution = phase_distributions[phase]

    # Number of batches available per ELO range
    available_batches = {
        "2000-2199": 30,  # 30 batches of ~25k positions each
        "2200-2399": 20,  # 20 batches of ~25k positions each
        "2400-2599": 5,  # 5 batches of ~25k positions each
    }

    # Calculate how many batches to use from each range
    # For standard phases, use ~30 batches total (increased from 25)
    total_batches_target = 30  # Increased from 25 for more data
    if curriculum_config and "total_batches" in curriculum_config:
        total_batches_target = curriculum_config["total_batches"]

    batches_to_use = {}

    for elo_range, percentage in distribution.items():
        batches_to_use[elo_range] = min(
            round(total_batches_target * percentage),
            available_batches[elo_range]
        )

    print(f"\nPhase {phase} Dataset Configuration:")
    print(f"--------------------------------")

    # Load datasets for each ELO range
    datasets = []
    folders_used = []

    # Use three months of data (extended from original code)
    for year_month in ["2016_07", "2016_08", "2016_09"]:  # Added an extra month
        for elo_range, num_batches in batches_to_use.items():
            if num_batches <= 0:
                continue

            folder_name = f"{year_month}_{elo_range}"
            folder_path = os.path.join(data_dir, folder_name)

            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} doesn't exist, skipping")
                continue

            # Get available batch files in this folder
            batch_files = [f for f in os.listdir(folder_path) if f.startswith("batch_") and f.endswith(".pkl")]

            if not batch_files:
                print(f"Warning: No batch files found in {folder_path}, skipping")
                continue

            # Select random batches for this folder (a third of the requested number)
            batches_per_folder = min(num_batches // 3, len(batch_files))

            # Apply data selection strategy from curriculum config if available
            if curriculum_config and curriculum_config.get("data_selection") == "hardest_positions":
                # In this hypothetical example, we'd select batches with hardest positions
                # In practice, you'd need a way to rank batches by difficulty
                selected_batches = batch_files[-batches_per_folder:]  # Placeholder selection
            else:
                # Default: random selection
                selected_batches = random.sample(batch_files, batches_per_folder)

            folders_used.append(f"{folder_name} ({batches_per_folder} batches)")

            print(f"Loading {batches_per_folder} randomly selected batches from {folder_name}:")
            for batch in selected_batches:
                print(f"  - {batch}")

            # Create individual datasets for each selected batch
            for batch_file in selected_batches:
                batch_path = os.path.join(folder_path, batch_file)
                # Create dataset with just this specific batch file
                dataset = NNUEDataset(
                    data_dir=folder_path,
                    batch_pattern=batch_file,  # Use exact batch filename
                    cache_size=2500  # Increased from 2000
                )
                datasets.append(dataset)

    # Summarize data usage
    print(f"\nData sources: {', '.join(folders_used)}")

    # Calculate total positions
    total_positions = sum(len(ds) for ds in datasets)
    print(f"Total positions: {total_positions:,}")

    if not datasets:
        raise ValueError("No datasets loaded. Check your data paths.")

    # Combine all datasets
    combined_dataset = ConcatDataset(datasets)

    # Apply curriculum filtering if specified
    if curriculum_config and curriculum_config.get("filter_positions"):
        # Implement position filtering based on curriculum needs
        # This is a placeholder for position-specific filtering logic
        print("Applying position filtering for curriculum learning")
        # combined_dataset = filter_positions_by_criteria(combined_dataset, curriculum_config)

    # Split into train, validation, and test
    val_size = int(0.1 * len(combined_dataset))
    test_size = int(0.05 * len(combined_dataset))
    train_size = len(combined_dataset) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        combined_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train positions: {len(train_dataset):,}")
    print(f"Validation positions: {len(val_dataset):,}")
    print(f"Test positions: {len(test_dataset):,}")

    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers):
    """Create data loaders for training, validation, and testing"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_phase_config(phase, previous_model_path=None):
    """Get training configuration for each phase with optimized hyperparameters"""

    # Base optimized values (from Optuna) - adjusted from the original
    optimized_values = {
        "hidden_size": 384,
        "learning_rate": 0.00406568155866974,
        "batch_size": 1024,
        "weight_decay": 1.0895579988345462e-06,
        "dropout_rate": 0.3880649861019966
    }

    # Advanced training features
    advanced_features = {
        "use_amp": True,  # Mixed precision training
        "gradient_clip": 0.5,  # Gradient clipping
        "label_smoothing": 0.1,  # Label smoothing
        "optimizer": "adamw",  # AdamW optimizer
        "score_scale": 1000.0,  # Score scaling
        "save_freq": 1  # Save frequency
    }

    # Enhanced 5-phase training strategy with refined hyperparameters
    configs = {
        # Phase 1: Foundation - establish basic evaluation patterns
        1: {
            "hidden_size": optimized_values["hidden_size"],
            "learning_rate": optimized_values["learning_rate"] * 0.8,  # Slightly reduced for stable start
            "weight_decay": optimized_values["weight_decay"] * 0.8,  # Slightly reduced for more exploration
            "dropout_rate": optimized_values["dropout_rate"] * 1.1,  # Slightly increased for regularization
            "batch_size": optimized_values["batch_size"],
            "epochs": 30,  # Extended training time
            "scheduler": "one_cycle",  # One cycle learning rate
            "early_stopping": 10,  # More patience
            "warmup_epochs": 5,  # Extended warmup
            "auxiliary_losses": False,  # No auxiliary losses in first phase
        },

        # Phase 2: General Patterns - build on foundation
        2: {
            "hidden_size": optimized_values["hidden_size"],
            "learning_rate": optimized_values["learning_rate"] * 0.4,  # Reduced rate
            "weight_decay": optimized_values["weight_decay"] * 2,  # Increased regularization
            "dropout_rate": optimized_values["dropout_rate"] * 0.9,  # Slightly reduced dropout
            "batch_size": optimized_values["batch_size"],
            "epochs": 25,  # Moderate training time
            "scheduler": "cosine",  # Switch to cosine annealing
            "early_stopping": 8,  # Moderate patience
            "warmup_epochs": 3,  # Moderate warmup
            "auxiliary_losses": True,  # Enable auxiliary losses
        },

        # Phase 3: Advanced Patterns - MODIFIED for better performance
        3: {
            "hidden_size": optimized_values["hidden_size"],
            "learning_rate": optimized_values["learning_rate"] * 0.25,  # Increased from 0.2
            "weight_decay": optimized_values["weight_decay"] * 3,  # Reduced from 4 for less regularization
            "dropout_rate": optimized_values["dropout_rate"] * 0.8,  # Increased from 0.7
            "batch_size": int(optimized_values["batch_size"] * 1.2),  # Slightly reduced from 1.25
            "epochs": 30,  # Increased from 25
            "scheduler": "cosine_warmup",  # Cosine with warmup
            "early_stopping": 12,  # Increased patience from 10
            "warmup_epochs": 3,  # Increased from 2
            "auxiliary_losses": True,  # Enable auxiliary losses
        },

        # Phase 4: Expert Knowledge - MODIFIED with improved parameters
        4: {
            "hidden_size": optimized_values["hidden_size"],
            "learning_rate": optimized_values["learning_rate"] * 0.15,  # Increased from 0.1
            "weight_decay": optimized_values["weight_decay"] * 6,  # Reduced from 8
            "dropout_rate": optimized_values["dropout_rate"] * 0.6,  # Increased from 0.5
            "batch_size": int(optimized_values["batch_size"] * 1.25),  # Reduced from 1.5
            "epochs": 25,  # Increased from 20
            "scheduler": "cosine_restarts",  # Cosine with restarts
            "early_stopping": 12,  # Unchanged
            "warmup_epochs": 2,  # Increased from 1
            "auxiliary_losses": True,  # Enable auxiliary losses
        },

        # Phase 5: Fine-tuning - MODIFIED with less aggressive parameters
        5: {
            "hidden_size": optimized_values["hidden_size"],
            "learning_rate": optimized_values["learning_rate"] * 0.08,  # Increased from 0.05
            "weight_decay": optimized_values["weight_decay"] * 8,  # Reduced from 12
            "dropout_rate": optimized_values["dropout_rate"] * 0.4,  # Increased from 0.3
            "batch_size": int(optimized_values["batch_size"] * 1.5),  # Reduced from 2.0
            "epochs": 20,  # Increased from 15
            "scheduler": "cosine",  # Back to standard cosine
            "early_stopping": 15,  # Maximum patience
            "warmup_epochs": 1,  # Added 1 epoch warmup (was 0)
            "auxiliary_losses": True,  # Enable auxiliary losses
        }
    }

    # Add specialized phases if needed

    # Get the base config for this phase
    config = configs[phase]

    # Add common configuration parameters
    config.update({
        "phase": phase,
        "pretrained_model": previous_model_path,
        # Modified output directory to include timestamp for better tracking
        "output_dir": os.path.join("models", f"phase_{phase}", f"run_{time.strftime('%Y%m%d_%H%M%S')}"),
        "val_split": 0.1,
        "num_workers": 6,
        "mixup_alpha": 0.15 if phase > 1 else 0.0,  # Enable mixup after phase 1, increased from 0.1
        "stochastic_depth_prob": 0.15 if phase > 3 else 0.0,
        # Enable stochastic depth in later phases, increased from 0.1
    })

    # Add advanced training features
    config.update(advanced_features)

    return config


def get_curriculum_config(phase):
    """
    Get curriculum learning configuration for specialized training phases
    These configurations can customize dataset loading and training focus
    """
    # No special curriculum for standard phases
    if phase <= 5:
        return None

    # Example specialized curriculum configurations
    curriculum_configs = {
        # Phase 6: Specialized tactical training (hypothetical)
        6: {
            "name": "tactical_focus",
            "description": "Focus on tactical positions",
            "distribution_override": {
                "2000-2199": 0.2,
                "2200-2399": 0.3,
                "2400-2599": 0.5
            },
            "total_batches": 20,
            "data_selection": "tactical_positions",
            "filter_positions": True,
        },

        # Phase 7: Specialized endgame training (hypothetical)
        7: {
            "name": "endgame_focus",
            "description": "Focus on endgame positions",
            "distribution_override": {
                "2000-2199": 0.1,
                "2200-2399": 0.3,
                "2400-2599": 0.6
            },
            "total_batches": 15,
            "data_selection": "endgame_positions",
            "filter_positions": True,
        }
    }

    return curriculum_configs.get(phase)


def train_phase(phase, data_dir, previous_model_path=None):
    """Train a single phase of the multi-phase training process"""

    print(f"\n{'=' * 60}")
    phase_names = {
        1: "Foundation",
        2: "General Patterns",
        3: "Advanced Patterns",
        4: "Expert Knowledge",
        5: "Fine-tuning",
        6: "Tactical Specialization",
        7: "Endgame Specialization"
    }
    phase_name = phase_names.get(phase, f"Phase {phase}")
    print(f"PHASE {phase}: {phase_name}")
    print(f"{'=' * 60}")

    # Get phase-specific configuration
    config = get_phase_config(phase, previous_model_path)

    # Get curriculum configuration if applicable
    curriculum_config = get_curriculum_config(phase)
    if curriculum_config:
        print(f"Using curriculum: {curriculum_config['name']} - {curriculum_config['description']}")

    # Update training status to in_progress
    save_training_status(
        os.path.dirname(os.path.dirname(config['output_dir'])),  # Base models directory
        phase,
        "in_progress",
        previous_model_path
    )

    try:
        # Set up datasets for this phase with potential curriculum
        train_dataset, val_dataset, test_dataset = setup_datasets(
            data_dir,
            phase,
            curriculum_config
        )

        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset,
            config['batch_size'], config['num_workers']
        )

        # Make sure the output directory exists
        os.makedirs(config['output_dir'], exist_ok=True)
        print(f"Models for this phase will be saved to: {config['output_dir']}")

        # Initialize model
        if previous_model_path and os.path.exists(previous_model_path):
            print(f"Loading pretrained model from {previous_model_path}")
            model = NNUE.load(previous_model_path)

            # Adjust dropout rate for current phase
            if hasattr(model, 'dropout') and model.dropout.p != config['dropout_rate']:
                print(f"Updating dropout rate from {model.dropout.p} to {config['dropout_rate']}")
                model.dropout.p = config['dropout_rate']

            # Potential model architecture modifications for specialized phases
            if phase > 5 and hasattr(model, 'modify_for_specialization'):
                print(f"Modifying model architecture for specialization phase {phase}")
                model.modify_for_specialization(phase)

        else:
            print(f"Creating new model with hidden size: {config['hidden_size']}")
            model = NNUE(
                hidden_size=config['hidden_size'],
                dropout_rate=config['dropout_rate']
            )

        # Create trainer with advanced features
        trainer = Trainer(config)

        # Train model
        results = trainer.train(model, train_loader, val_loader, test_loader)

        print(f"\nPhase {phase} training complete!")
        print(f"Best validation loss: {results['best_val_loss']:.6f}")
        print(f"Best validation accuracy: {results['best_accuracy']:.4f}")

        # Perform additional validation if desired
        # if phase >= 3:
        #    print("Performing additional specialized validation...")
        #    specialized_accuracy = evaluate_on_specialized_positions(model)
        #    print(f"Specialized validation accuracy: {specialized_accuracy:.4f}")

        # Update training status to completed
        save_training_status(
            os.path.dirname(os.path.dirname(config['output_dir'])),  # Base models directory
            phase,
            "completed",
            results['model_path'],
            {
                "best_val_loss": float(results['best_val_loss']),
                "best_accuracy": float(results['best_accuracy'])
            }
        )

        return results['model_path']

    except Exception as e:
        # Update training status to failed
        save_training_status(
            os.path.dirname(os.path.dirname(config['output_dir'])),  # Base models directory
            phase,
            "failed",
            None,
            {
                "error": str(e)
            }
        )
        raise  # Re-raise the exception to be handled by the main function


def main():
    """Main training pipeline with automatic checkpoint detection"""
    print("\n" + "=" * 80)
    print("NNUE CHESS EVALUATION NEURAL NETWORK TRAINING")
    print("=" * 80)

    # Configuration
    data_dir = "../data"  # Base directory for training data
    output_dir = "models"  # Directory to save models
    seed = 42  # Random seed for reproducibility
    total_phases = 5  # Total number of phases in standard training

    # Print hardware information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")
    if device.type == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    print(f"PyTorch Version: {torch.__version__}")

    # Set seeds for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True  # Performance optimization

    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect which phase to start from
    start_phase, model_path = detect_training_phase(output_dir, total_phases)

    # Create phase-specific directories
    for phase in range(1, total_phases + 1):
        phase_dir = os.path.join(output_dir, f"phase_{phase}")
        os.makedirs(phase_dir, exist_ok=True)

    # Execute training phases
    phase_results = {}
    for phase in range(start_phase, total_phases + 1):
        # Train this phase
        phase_start_time = time.time()
        try:
            model_path = train_phase(
                phase,
                data_dir,
                model_path
            )
            phase_duration = time.time() - phase_start_time
            phase_results[phase] = {
                "model_path": model_path,
                "duration": phase_duration,
                "status": "completed"
            }
            print(f"Phase {phase} completed in {phase_duration / 60:.2f} minutes")
        except Exception as e:
            phase_duration = time.time() - phase_start_time
            phase_results[phase] = {
                "model_path": None,
                "duration": phase_duration,
                "status": "failed",
                "error": str(e)
            }
            print(f"Error in Phase {phase}: {str(e)}")
            print(f"Training interrupted after {phase_duration / 60:.2f} minutes")
            break

        # Save intermediate evaluation results between phases
        print(f"Saving detailed evaluation for Phase {phase}...")
        evaluation_path = os.path.join(output_dir, f"phase_{phase}_evaluation.txt")
        with open(evaluation_path, "w") as f:
            f.write(f"Phase {phase} completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training duration: {phase_duration:.2f} seconds ({phase_duration / 60:.2f} minutes)\n")
            f.write(f"Model path: {model_path}\n\n")
            # Add more evaluation metrics as needed

    # Print summary of all phases
    print("\nTraining Summary:")
    print("=" * 60)
    for phase, result in phase_results.items():
        status = result["status"]
        duration = result["duration"] / 60
        model_info = f"model: {os.path.basename(result['model_path'])}" if result["model_path"] else "no model saved"
        print(f"Phase {phase}: {status}, {duration:.2f} minutes, {model_info}")

    successful_phases = [p for p, r in phase_results.items() if r["status"] == "completed"]
    if successful_phases:
        total_time = sum(result['duration'] for result in phase_results.values())
        print(f"Total training time: {total_time / 60:.2f} minutes ({total_time / 3600:.2f} hours)")

        if successful_phases[-1] == total_phases:
            print("\nTraining successfully completed all phases!")
        else:
            print(f"\nTraining completed through phase {successful_phases[-1]}.")


if __name__ == "__main__":
    main()

