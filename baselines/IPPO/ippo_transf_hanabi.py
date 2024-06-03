"""
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import distrax
import wandb
import functools
import matplotlib.pyplot as plt
import hydra
import jaxmarl

from flax.linen.initializers import constant, orthogonal
from typing import List, Sequence, NamedTuple, Any, Dict, Tuple
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.wrappers.hanabi_token_obs import HanabiTokenObsWrapper

# TODO:
# - Are the batchify and unbatchify functions jitted?


def get_activation(activation_name):
    if activation_name == "relu":
        return nn.relu
    elif activation_name == "tanh":
        return nn.tanh
    else:
        raise ValueError(f"Unknown activation: {activation_name}")


class EncoderBlock(nn.Module):
    # Input dimension is needed here since it is equal to the output dimension (residual
    # connection):
    hidden_dim: int = 32
    num_heads: int = 4
    dim_feedforward: int = 256
    dropout_prob: float = 0.0

    def setup(self):
        # Attention layer
        self.self_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_prob,
            kernel_init=nn.initializers.xavier_uniform(),
            use_bias=False,
        )
        # Two-layer MLP
        self.linear = [
            nn.Dense(
                self.dim_feedforward,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=constant(0.0),
            ),
            nn.Dense(
                self.hidden_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=constant(0.0),
            ),
        ]
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, mask=None, deterministic=True):

        # Attention part
        # masking is not compatible with fast self attention
        if mask is not None and not self.use_fast_attention:
            mask = jnp.repeat(
                nn.make_attention_mask(mask, mask), self.num_heads, axis=-3
            )
        attended = self.self_attn(
            inputs_q=x, inputs_kv=x, mask=mask, deterministic=deterministic
        )

        x = self.norm1(attended + x)
        x = x + self.dropout(x, deterministic=deterministic)

        # MLP part
        feedforward = self.linear[0](x)
        feedforward = nn.relu(feedforward)
        feedforward = self.linear[1](feedforward)

        x = self.norm2(feedforward + x)
        x = x + self.dropout(x, deterministic=deterministic)

        return x


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    transf_layers: int = 2

    @nn.compact
    def __call__(self, x):
        obs, dones, avail_actions = x
        obs_embedding = nn.Dense(
            32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        obs_embedding = nn.relu(obs_embedding)

        actions_embeddings = nn.Embed(num_embeddings=self.action_dim, features=32)(
            jnp.arange(self.action_dim)
        )
        actions_embeddings = jnp.tile(
            actions_embeddings, (*obs_embedding.shape[:-1], 1, 1)
        )
        embeddings = jnp.concatenate(
            (obs_embedding[..., np.newaxis, :], actions_embeddings), axis=-2
        )

        for _ in range(self.transf_layers):
            embeddings = EncoderBlock()(embeddings)

        obs_embedding_post = embeddings[..., 0, :]
        actions_embeddings_post = embeddings[..., 1:, :]

        # actor_mean = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(actions_embeddings_post)
        # actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(
            actions_embeddings_post
        )
        actor_mean = jnp.squeeze(actor_mean, axis=-1)

        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            obs_embedding_post
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class SingleObsTransformerActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    num_obs_sequences: int
    # TODO: Remove defaults:
    num_players: int = 2
    num_colors: int = 5
    hand_size: int = 5

    transf_layers: int = 2
    activation: str = "tanh"
    hidden_size: int = 64
    embed_dim: int = 32
    hand_size: int = 5
    ff_dim: int = 64
    dropout_prob: float = 0.0

    @nn.compact
    def __call__(self, x):
        *obs, dones, avail_actions = x
        activation = get_activation(self.activation)
        print(f"{obs=}")
        # Different dense encoder for each sequence in obs:
        obs_embeddings = [
            nn.Dense(self.embed_dim)(obs[i]) for i in range(self.num_obs_sequences)
        ]
        obs_embeddings = jnp.concatenate(obs_embeddings, axis=-2)
        print(f"{obs_embeddings=}")
        # TODO: Try activation here:
        # obs_embeddings = activation(obs_embeddings)

        # Append a learnable 'no-op' and 'value' token. These will be used to decode
        # the logits for the no-op action and the critic value:
        extra_embeddings = nn.Embed(num_embeddings=2, features=self.embed_dim)(
            jnp.array((0, 1), dtype=int)
        )
        # Repeat the extra tokens to match the batch size:
        extra_embeddings = jnp.tile(
            extra_embeddings, (*obs_embeddings.shape[:-2], 1, 1)
        )
        print(f"{extra_embeddings=}")
        obs_embeddings = jnp.concatenate([obs_embeddings, extra_embeddings], axis=-2)
        print(f"{obs_embeddings=}")

        for _ in range(self.transf_layers):
            obs_embeddings = EncoderBlock(
                hidden_dim=self.embed_dim,
                dim_feedforward=self.ff_dim,
                dropout_prob=self.dropout_prob,
            )(obs_embeddings)

        # Extract the actionable cards:
        # TODO: This will be replaced with a mask computed elsewhere:
        num_hand_cards = self.num_players * self.hand_size
        # There are num_obs_sequences extra elements, because currently all
        # extra sequences only have one element (minus two for value and no-op).
        # There are num_colors fireworks tokens, and two extra tokens at the end of the
        # sequence. Before that are the hand cards:
        num_trailing = self.num_obs_sequences - 1 + self.num_colors + 2

        # Only keep encodings for the newest observation:
        obs_embeddings = obs_embeddings[..., -1, :, :]
        print(f"{obs_embeddings=}")

        # First token is action token, then own hand, next player hand, etc.; last two
        # tokens are no-op and value:
        action_embeddings = jnp.concatenate(
            (
                # obs_embeddings[..., 1 : num_hand_cards + 1, :],
                # TODO: This is a temporary hack to get the card indices from the end:
                obs_embeddings[..., -num_trailing - num_hand_cards : -num_trailing, :],
                obs_embeddings[..., -2:, :],
            ),
            axis=-2,
        )
        print(f"{action_embeddings=}")

        # Two actions per card (remove value token):
        actor_mean = nn.Dense(2)(action_embeddings[..., :-1, :])
        # Flatten last two dims into action vector:
        actor_mean = actor_mean.reshape(*actor_mean.shape[:-2], -1)
        # No-op action gets decoded into two values, so remove last action:
        actor_mean = actor_mean[..., :-1]

        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        # Decode value from the last action embedding (the "value" token):
        value_embedding = action_embeddings[..., -1, :]
        critic = nn.Dense(512)(value_embedding)
        critic = activation(critic)
        critic = nn.Dense(1)(critic)
        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: Tuple[jnp.ndarray, ...]
    # obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def batchify_obs(x: dict, agent_list, num_actors, sequence_length):
    # TODO: Improve:
    num_sequences = len(x[agent_list[0]])
    x = [jnp.vstack([x[a][seq] for a in agent_list]) for seq in range(num_sequences)]
    return tuple(
        # TODO: *x[seq].shape[-2:] if not using sequences:
        x[seq].reshape((num_actors, *x[seq].shape[-3:]))
        for seq in range(num_sequences)
    )


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = HanabiTokenObsWrapper(env, **config["TOKEN_WRAPPER_KWARGS"])
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)

        # INIT NETWORK
        network = SingleObsTransformerActorCritic(
            env.action_space(env.agents[0]).n,
            config=config,
            num_obs_sequences=len(obsv[env.agents[0]]),
            **config["MODEL_KWARGS"],
        )
        # network = ActorCritic(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)

        # Add sequence dimension to obsv:
        obsv = jax.tree_map(lambda x: x[:, np.newaxis, :], obsv)

        # Use env obs to initialize network:
        init_x = (
            *batchify_obs(
                obsv, env.agents, config["NUM_ACTORS"], config["SEQUENCE_LENGTH"]
            ),
            jnp.zeros((1, config["NUM_ACTORS"])),
            jnp.zeros((1, config["NUM_ACTORS"], env.action_space(env.agents[0]).n)),
        )
        network_params = network.init(_rng, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, rng = runner_state

                # SELECT ACTION
                avail_actions = jax.vmap(env.get_legal_moves)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )

                obs_batch = batchify_obs(
                    last_obs,
                    env.agents,
                    config["NUM_ACTORS"],
                    config["SEQUENCE_LENGTH"],
                )

                # obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    *obs_batch,
                    # obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions[np.newaxis, :],
                )

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = jax.tree_map(lambda x: x.squeeze(), env_act)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                # Shift last_obs back by one along sequence dim and insert new obs at
                # end:
                obsv = jax.tree_map(
                    lambda x, y: jnp.concatenate(
                        (x[:, 1:, ...], y[:, np.newaxis, ...]), axis=1
                    ),
                    last_obs,
                    obsv,
                )

                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    done_batch,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions,
                )
                runner_state = (train_state, env_state, obsv, done_batch, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, rng = runner_state
            last_obs_batch = batchify_obs(
                last_obs, env.agents, config["NUM_ACTORS"], config["SEQUENCE_LENGTH"]
            )
            avail_actions = jnp.ones(
                (config["NUM_ACTORS"], env.action_space(env.agents[0]).n)
            )
            ac_in = (
                *last_obs_batch,
                # last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
                avail_actions,
            )
            _, last_val = network.apply(train_state.params, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(
                            params,
                            (
                                *traj_batch.obs,
                                traj_batch.done,
                                traj_batch.avail_actions,
                            ),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch = (traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            def callback(metric):
                wandb.log(
                    {
                        "returns": metric["returned_episode_returns"][-1, :].mean(),
                        "env_step": metric["update_steps"]
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                    }
                )

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, rng)
            return (runner_state, update_steps), None

        # Add a sequence dimension to obsv:
        print(f"\t1: {obsv=}")
        # Left pad the sequence dimension with zeros to sequence length:
        obsv = jax.tree_map(
            lambda x: jnp.concatenate(
                (
                    jnp.zeros(
                        (*x.shape[:1], config["SEQUENCE_LENGTH"] - 1, *x.shape[2:])
                    ),
                    x,
                ),
                axis=1,
            ),
            obsv,
        )
        print(f"\t3: {obsv=}")

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            _rng,
        )
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


def init_wandb_run(config, trial_idx=None):
    group_name = config["WANDB_RUN_GROUP"]
    run_name = config["WANDB_RUN_NAME"]

    if trial_idx is not None:
        group_name = f"{group_name}_individual"
        run_name = f"{run_name}_{trial_idx}"

    return wandb.init(
        group=group_name,
        name=run_name,
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["PPO", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
        config=config,
        mode=config["WANDB_MODE"],
    )


@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_hanabi")
def main(config):
    config = OmegaConf.to_container(config)

    init_wandb_run(config)

    # TODO: Vectorisation is not working:
    # rng = jax.random.PRNGKey(config["SEED"])
    # rngs = jax.random.split(rng, config["NUM_SEEDS"])
    # device = jax.devices()[config["GPU_IDX"]]
    # train_vjit = jax.jit(jax.vmap(make_train(config)), device=device)
    # outs = jax.block_until_ready(train_vjit(rngs))

    rng = jax.random.PRNGKey(config["SEED"])
    device = jax.devices()[config["GPU_IDX"]]
    train_jit = jax.jit(make_train(config), device=device)
    out = train_jit(rng)


if __name__ == "__main__":
    # jax.config.update("jax_disable_jit", True)
    main()
    """results = out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1)
    jnp.save('hanabi_results', results)
    plt.plot(results)
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f'IPPO_{config["ENV_NAME"]}.png')"""
