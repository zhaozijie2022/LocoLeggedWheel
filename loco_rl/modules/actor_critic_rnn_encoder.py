from __future__ import annotations

import torch

from loco_rl.modules.actor_critic import ActorCritic
from loco_rl.models import MLP
from loco_rl.modules.actor_critic_recurrent import Memory
from loco_rl.utils import unpad_trajectories


class ActorCriticRNNEncoder(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        actor_obs_dim,
        critic_obs_dim,
        num_actions,
        actor_flatten_obs_end_idx,
        actor_encoder_obs_start_idx,
        actor_encoder_hidden_dims,
        actor_encoder_embedding_dim,
        actor_hidden_dims,
        critic_flatten_obs_end_idx,
        critic_encoder_obs_start_idx,
        critic_encoder_hidden_dims,
        critic_encoder_embedding_dim,
        critic_hidden_dims,
        encoder_rnn_type="gru",
        encoder_rnn_hidden_size=256,
        encoder_rnn_num_layers=1,
        encoder_activation="elu",
        encoder_final_activation=None,
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticEncoder.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        
        self.actor_encoder_obs_dim = abs(actor_encoder_obs_start_idx) if actor_encoder_obs_start_idx < 0 else (actor_obs_dim-actor_encoder_obs_start_idx)
        self.actor_flatten_obs_dim = actor_flatten_obs_end_idx if actor_flatten_obs_end_idx > 0 else (actor_obs_dim-abs(actor_flatten_obs_end_idx))

        self.critic_with_encoder = False if critic_encoder_hidden_dims is None else True
        if self.critic_with_encoder:
            assert critic_flatten_obs_end_idx is not None, "Critic flatten obs end index is required"
            assert critic_encoder_obs_start_idx is not None, "Critic encoder obs start index is required"
            assert critic_encoder_embedding_dim is not None, "Critic encoder embedding dim is required"
            self.critic_encoder_obs_dim = abs(critic_encoder_obs_start_idx) if critic_encoder_obs_start_idx < 0 else (critic_obs_dim-critic_encoder_obs_start_idx)
            self.critic_flatten_obs_dim = critic_flatten_obs_end_idx if critic_flatten_obs_end_idx > 0 else (critic_obs_dim-abs(critic_flatten_obs_end_idx))

        # actor critic backbone
        super().__init__(
            num_actor_obs=self.actor_flatten_obs_dim+actor_encoder_embedding_dim,
            num_critic_obs=(self.critic_flatten_obs_dim+critic_encoder_embedding_dim) if self.critic_with_encoder else critic_obs_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        # actor encoder memory
        self.memory_a = Memory(
            input_size=self.actor_encoder_obs_dim,
            type=encoder_rnn_type,
            num_layers=encoder_rnn_num_layers,
            hidden_size=encoder_rnn_hidden_size,
            )
        print(f"Actor RNN: {self.memory_a}")

        # actor encoder
        self.actor_encoder = MLP(
            encoder_rnn_hidden_size,
            actor_encoder_hidden_dims,
            actor_encoder_embedding_dim,
            activation=encoder_activation,
            final_layer_activation=encoder_final_activation,
            )
        print(f"Actor Encoder: {self.actor_encoder}")

        if self.critic_with_encoder:
            # critic encoder memory
            self.memory_c = Memory(
                input_size=self.critic_encoder_obs_dim,
                type=encoder_rnn_type,
                num_layers=encoder_rnn_num_layers,
                hidden_size=encoder_rnn_hidden_size,
                )
            print(f"Critic RNN: {self.memory_c}")

            # critic encoder
            self.critic_encoder = MLP(
                encoder_rnn_hidden_size,
                critic_encoder_hidden_dims,
                critic_encoder_embedding_dim,
                activation=encoder_activation,
                final_layer_activation=encoder_final_activation,
            )
            print(f"Critic Encoder: {self.critic_encoder}")

    def reset(self, dones=None):
        super().reset(dones)
        self.memory_a.reset(dones)
        if self.critic_with_encoder:
            self.memory_c.reset(dones)

    def act(self, obs, masks=None, hidden_states=None):
        flatten_obs = obs[..., :self.actor_flatten_obs_dim]
        if masks is not None: flatten_obs = unpad_trajectories(flatten_obs, masks)
        encoder_obs = obs[..., -self.actor_encoder_obs_dim:]
        encoder_obs = self.memory_a(encoder_obs, masks, hidden_states).squeeze(0)
        embedding = self.actor_encoder(encoder_obs)
        input_a = torch.cat([flatten_obs, embedding], dim=-1)
        return super().act(input_a)

    def act_inference(self, obs):
        flatten_obs = obs[..., :self.actor_flatten_obs_dim]
        encoder_obs = obs[..., -self.actor_encoder_obs_dim:]
        encoder_obs = self.memory_a(encoder_obs).squeeze(0)
        embedding = self.actor_encoder(encoder_obs)
        input_a = torch.cat([flatten_obs, embedding], dim=-1)
        return super().act_inference(input_a)
    
    def act_encoder_inference(self, encoder_obs):
        encoder_obs = self.memory_a(encoder_obs).squeeze(0)
        return self.actor_encoder(encoder_obs)
    
    def act_backbone_inference(self, flatten_obs, embedding):
        input_a = torch.cat([flatten_obs, embedding], dim=-1)
        return super().act_inference(input_a)

    def evaluate(self, obs, masks=None, hidden_states=None):
        if self.critic_with_encoder:
            flatten_obs = obs[..., :self.critic_flatten_obs_dim]
            if masks is not None: flatten_obs = unpad_trajectories(flatten_obs, masks)
            encoder_obs = obs[..., -self.critic_encoder_obs_dim:]
            encoder_obs = self.memory_c(encoder_obs, masks, hidden_states).squeeze(0)
            embedding = self.critic_encoder(encoder_obs)
            input_c = torch.cat([flatten_obs, embedding], dim=-1)
        else:
            input_c = obs
        return super().evaluate(input_c)

    def get_hidden_states(self):
        if self.critic_with_encoder:
            return self.memory_a.hidden_states, self.memory_c.hidden_states
        else:
            return self.memory_a.hidden_states, None


