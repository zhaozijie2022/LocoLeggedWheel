from __future__ import annotations
import torch
from loco_rl.modules.actor_critic import ActorCritic
from loco_rl.models import *


class ActorCriticEncoder(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        actor_obs_dim,
        critic_obs_dim,
        num_actions,
        # actor_encoder_obs_dim,
        actor_flatten_obs_end_idx,
        actor_encoder_obs_start_idx,
        actor_encoder_hidden_dims,
        actor_encoder_embedding_dim,
        actor_hidden_dims,
        # critic_encoder_obs_dim,
        critic_flatten_obs_end_idx,
        critic_encoder_obs_start_idx,
        critic_encoder_hidden_dims,
        critic_encoder_embedding_dim,
        critic_hidden_dims,
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

        super().__init__(
            num_actor_obs=self.actor_flatten_obs_dim+actor_encoder_embedding_dim,
            num_critic_obs=(self.critic_encoder_obs_dim+critic_encoder_embedding_dim) if self.critic_with_encoder else critic_obs_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        self.actor_encoder = MLP(
            self.actor_encoder_obs_dim,
            actor_encoder_hidden_dims,
            actor_encoder_embedding_dim,
            activation=encoder_activation,
            final_layer_activation=encoder_final_activation,
        )
        print(f"Actor Encoder: {self.actor_encoder}")

        if self.critic_with_encoder:
            self.critic_encoder = MLP(
                self.critic_encoder_obs_dim,
                critic_encoder_hidden_dims,
                critic_encoder_embedding_dim,
                activation=encoder_activation,
                final_layer_activation=encoder_final_activation,
            )
            print(f"Critic Encoder: {self.critic_encoder}")

    def act(self, obs, **kwargs):
        flatten_obs = obs[..., :self.actor_flatten_obs_dim]
        encoder_obs = obs[..., -self.actor_encoder_obs_dim:]
        embedding = self.actor_encoder(encoder_obs)
        input_a = torch.cat([flatten_obs, embedding], dim=-1)
        return super().act(input_a)

    def act_inference(self, obs):
        flatten_obs = obs[..., :self.actor_flatten_obs_dim]
        encoder_obs = obs[..., -self.actor_encoder_obs_dim:]
        embedding = self.actor_encoder(encoder_obs)
        input_a = torch.cat([flatten_obs, embedding], dim=-1)
        return super().act_inference(input_a)
    
    def act_encoder_inference(self, encoder_obs):
        return self.actor_encoder(encoder_obs)
    
    def act_backbone_inference(self, flatten_obs, embedding):
        input_a = torch.cat([flatten_obs, embedding], dim=-1)
        return super().act_inference(input_a)

    def evaluate(self, obs, **kwargs):
        if self.critic_with_encoder:
            flatten_obs = obs[..., :self.critic_flatten_obs_dim]
            encoder_obs = obs[..., -self.critic_encoder_obs_dim:]
            embedding = self.critic_encoder(encoder_obs)
            input_c = torch.cat([flatten_obs, embedding], dim=-1)
        else:
            input_c = obs
        return super().evaluate(input_c)

    def get_actor_critic_obs_from_obs_dict(self, obs_dict):
        actor_obs = obs_dict["policy"]
        critic_obs = obs_dict.get("critic", actor_obs)
        encoder_obs = obs_dict["encoder"]
        actor_obs = torch.cat((actor_obs, encoder_obs), dim=1)
        critic_obs = torch.cat((critic_obs, encoder_obs), dim=1) if self.critic_with_encoder else critic_obs
        return actor_obs, critic_obs


