import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Logger pour suivre l'entraînement avec TensorBoard et logs texte
    """

    def __init__(self, log_dir='./logs', use_tensorboard=True, use_wandb=False, wandb_config=None):
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

        # Créer un dossier de logs avec timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(log_dir, f'run_{timestamp}')
        os.makedirs(self.run_dir, exist_ok=True)

        # TensorBoard
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.run_dir)

        # Weights & Biases
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_config.get('project', 'rocket-league-bot'),
                    config=wandb_config.get('config', {}),
                    dir=self.run_dir
                )
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, disabling W&B logging")
                self.use_wandb = False

        # Fichier de log texte
        self.log_file = os.path.join(self.run_dir, 'training.log')

    def log_scalar(self, tag, value, step):
        """Log une valeur scalaire"""
        if self.use_tensorboard:
            self.writer.add_scalar(tag, value, step)

        if self.use_wandb:
            self.wandb.log({tag: value}, step=step)

    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log plusieurs valeurs scalaires"""
        for tag, value in tag_scalar_dict.items():
            self.log_scalar(f'{main_tag}/{tag}', value, step)

    def log_text(self, message):
        """Log un message texte"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f'[{timestamp}] {message}'

        # Afficher dans la console
        print(log_message)

        # Écrire dans le fichier
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')

    def log_histogram(self, tag, values, step):
        """Log un histogramme"""
        if self.use_tensorboard:
            self.writer.add_histogram(tag, values, step)

    def close(self):
        """Ferme les loggers"""
        if self.use_tensorboard:
            self.writer.close()

        if self.use_wandb:
            self.wandb.finish()
