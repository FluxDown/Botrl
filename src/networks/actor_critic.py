import torch
import torch.nn as nn
import numpy as np


class ActorCritic(nn.Module):
    """
    Réseau de neurones Actor-Critic pour PPO

    - Actor: Produit une distribution de probabilités sur les actions
    - Critic: Estime la value function V(s)

    Architecture avec couches séparées pour actor et critic
    """

    def __init__(
        self,
        obs_space_size: int,
        action_space_size: int,
        policy_layers=[512, 512, 512],
        value_layers=[512, 512, 512],
        activation="relu",
        continuous_actions=True
    ):
        super(ActorCritic, self).__init__()

        self.obs_space_size = obs_space_size
        self.action_space_size = action_space_size
        self.continuous_actions = continuous_actions

        # Fonction d'activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

        # === Réseau Actor (Policy) ===
        policy_network = []
        input_size = obs_space_size

        for hidden_size in policy_layers:
            policy_network.append(nn.Linear(input_size, hidden_size))
            policy_network.append(nn.LayerNorm(hidden_size))
            policy_network.append(self.activation)
            input_size = hidden_size

        self.policy_net = nn.Sequential(*policy_network)

        # Couche de sortie pour l'actor
        if continuous_actions:
            # Pour actions continues: moyenne et log_std
            self.action_mean = nn.Linear(input_size, action_space_size)
            self.action_log_std = nn.Linear(input_size, action_space_size)
        else:
            # Pour actions discrètes: logits
            self.action_logits = nn.Linear(input_size, action_space_size)

        # === Réseau Critic (Value) ===
        value_network = []
        input_size = obs_space_size

        for hidden_size in value_layers:
            value_network.append(nn.Linear(input_size, hidden_size))
            value_network.append(nn.LayerNorm(hidden_size))
            value_network.append(self.activation)
            input_size = hidden_size

        self.value_net = nn.Sequential(*value_network)
        self.value_head = nn.Linear(input_size, 1)

        # Initialisation des poids
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialise les poids du réseau avec orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Initialisation spéciale pour les couches de sortie
        if self.continuous_actions:
            nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
            nn.init.orthogonal_(self.action_log_std.weight, gain=0.01)
        else:
            nn.init.orthogonal_(self.action_logits.weight, gain=0.01)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs):
        """
        Forward pass du réseau

        Returns:
            - action_dist: Distribution sur les actions
            - value: Estimation de la value function
        """
        # Actor
        policy_features = self.policy_net(obs)

        if self.continuous_actions:
            action_mean = self.action_mean(policy_features)
            action_log_std = self.action_log_std(policy_features)
            action_std = torch.exp(action_log_std)
            action_dist = torch.distributions.Normal(action_mean, action_std)
        else:
            action_logits = self.action_logits(policy_features)
            action_dist = torch.distributions.Categorical(logits=action_logits)

        # Critic
        value_features = self.value_net(obs)
        value = self.value_head(value_features)

        return action_dist, value

    def get_action(self, obs, deterministic=False):
        """
        Sélectionne une action basée sur l'observation

        Args:
            obs: Observation
            deterministic: Si True, prend l'action la plus probable (pour évaluation)

        Returns:
            - action: Action sélectionnée
            - log_prob: Log probabilité de l'action
            - value: Estimation de la value
        """
        action_dist, value = self.forward(obs)

        if deterministic:
            if self.continuous_actions:
                action = action_dist.mean
            else:
                action = action_dist.probs.argmax(dim=-1)
        else:
            action = action_dist.sample()

        log_prob = action_dist.log_prob(action)

        # Pour les actions continues multidimensionnelles, sommer les log_probs
        if self.continuous_actions and len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=-1)

        return action, log_prob, value

    def get_value(self, obs):
        """
        Calcule SEULEMENT la value (pour bootstrap, plus rapide).

        Args:
            obs: Observation (batch)

        Returns:
            value: Estimation de la value function
        """
        value_features = self.value_net(obs)
        value = self.value_head(value_features)
        return value

    def evaluate_actions(self, obs, actions):
        """
        Évalue les actions pour le training PPO

        Returns:
            - log_prob: Log probabilités des actions
            - entropy: Entropie de la distribution
            - value: Estimation de la value
        """
        action_dist, value = self.forward(obs)

        log_prob = action_dist.log_prob(actions)

        # Pour les actions continues multidimensionnelles
        if self.continuous_actions and len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=-1)

        entropy = action_dist.entropy()

        # Pour les actions continues multidimensionnelles
        if self.continuous_actions and len(entropy.shape) > 1:
            entropy = entropy.sum(dim=-1)

        return log_prob, entropy, value


class LSTMActorCritic(nn.Module):
    """
    Version avec LSTM pour capturer les dépendances temporelles
    """

    def __init__(
        self,
        obs_space_size: int,
        action_space_size: int,
        hidden_size=512,
        lstm_hidden_size=256,
        num_layers=2,
        continuous_actions=True
    ):
        super(LSTMActorCritic, self).__init__()

        self.obs_space_size = obs_space_size
        self.action_space_size = action_space_size
        self.continuous_actions = continuous_actions
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers

        # Encodeur d'observation
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_space_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )

        # LSTM partagé
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Actor head
        if continuous_actions:
            self.action_mean = nn.Linear(lstm_hidden_size, action_space_size)
            self.action_log_std = nn.Linear(lstm_hidden_size, action_space_size)
        else:
            self.action_logits = nn.Linear(lstm_hidden_size, action_space_size)

        # Critic head
        self.value_head = nn.Linear(lstm_hidden_size, 1)

    def forward(self, obs, hidden_state=None):
        """Forward avec LSTM"""
        batch_size = obs.shape[0]

        # Encoder l'observation
        features = self.obs_encoder(obs)

        # Ajouter dimension temporelle si nécessaire
        if len(features.shape) == 2:
            features = features.unsqueeze(1)

        # LSTM
        lstm_out, hidden_state = self.lstm(features, hidden_state)
        lstm_out = lstm_out[:, -1, :]  # Prendre la dernière sortie

        # Actor
        if self.continuous_actions:
            action_mean = self.action_mean(lstm_out)
            action_log_std = self.action_log_std(lstm_out)
            action_std = torch.exp(action_log_std)
            action_dist = torch.distributions.Normal(action_mean, action_std)
        else:
            action_logits = self.action_logits(lstm_out)
            action_dist = torch.distributions.Categorical(logits=action_logits)

        # Critic
        value = self.value_head(lstm_out)

        return action_dist, value, hidden_state

    def init_hidden_state(self, batch_size=1, device='cpu'):
        """Initialise l'état caché du LSTM"""
        return (
            torch.zeros(self.num_layers, batch_size, self.lstm_hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.lstm_hidden_size).to(device)
        )
