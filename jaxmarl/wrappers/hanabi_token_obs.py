import chex
import jax
import jax.numpy as jnp

from jax import lax, nn
from functools import partial
from typing import Tuple, Dict
from pprint import pprint

from jaxmarl.environments.hanabi.hanabi_game import State
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.wrappers.baselines import JaxMARLWrapper

# Ideas for future observations:
# - n-hot vector for e.g. hints (position of each affected card)
# - Sinusoidal encoding for ranks, bombs, deck size, hint tokens


class HanabiTokenObsWrapper(JaxMARLWrapper):
    # TODO:
    # - Non-card tokens
    # - Any action transformations
    # - Check if stop gradient necessary here

    def __init__(self, env: MultiAgentEnv, flatten_obs=False, use_token_actions=True):
        super().__init__(env)

        self.flatten_obs = flatten_obs
        self.use_token_actions = use_token_actions

        # TODO: Set obs and action spaces:
        card_enc_length = (self.num_colors + 1) + (self.num_ranks + 2)
        player_enc_length = self.num_agents + 1
        position_enc_length = self.hand_size + 2
        type_enc_length = self.num_agents + 3
        self.token_length = (
            card_enc_length + player_enc_length + position_enc_length + type_enc_length
        )

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, Tuple[chex.Array]], State]:
        obs, state = self._env.reset(key)
        obs = self.state_to_obs(True, None, state, None)
        return obs, state

    @partial(jax.jit, static_argnums=[0])
    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        action: Dict[str, chex.Array],
    ) -> Tuple[
        Dict[str, Tuple[chex.Array]], State, Dict[str, float], Dict[str, bool], Dict
    ]:
        if self.use_token_actions:
            action = self.process_actions(state, action)

        # TODO: how much performance improvement would there be if we skip creation
        # of the default observations?
        old_state = state
        obs, new_state, reward, done, info = self._env.step(key, old_state, action)

        obs = self.state_to_obs(False, old_state, new_state, action)
        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=[0])
    def process_actions(self, state, actions):
        print(f"BEFORE {actions=}")
        actions = {
            name: self.action_to_env_action(state, aidx, actions[name])
            for aidx, name in enumerate(self.agents)
        }
        print(f"AFTER {actions=}")
        return actions

    @partial(jax.jit, static_argnums=[0])
    def action_to_env_action(self, state, aidx, action):
        """
        Convert action from the "token order" (i.e. actions are ordered by the
        corresponding cards in the observation) to the order expected by the
        environment.
        """

        def convert_discard_play_action():
            # Token actions interleave discard, play for each card in hand.
            # Env actions: First 5 env actions are discards, second 5 are plays.
            card_idx = action // 2
            is_play = action % 2
            # Plays are shifted up hand_size:
            return card_idx + is_play * self.hand_size

        def convert_hint_action():
            # Actions are card indices, but need to be converted to rank or color hints.
            hands_from_self = jnp.roll(state.player_hands, -aidx, axis=0)
            teammate_hands = hands_from_self[1:]

            # Remove discard and play actions:
            hint_action = action - (2 * self.hand_size)
            # There are 2 * hand_size hint actions per teammate:
            teammate_idx = hint_action // (2 * self.hand_size)
            card_idx = (hint_action % (2 * self.hand_size)) // 2
            # Hints alternate color-rank:
            is_rank_hint = hint_action % 2 == 1

            # Get card corresponding to hint:
            hint_card = teammate_hands[teammate_idx, card_idx]
            hint_card_rank = hint_card.sum(axis=0)
            hint_card_colour = hint_card.sum(axis=1)
            hint_one_hot = lax.select(is_rank_hint, hint_card_rank, hint_card_colour)

            # Shift up by num_ranks for colour hints:
            hint_value = hint_one_hot.argmax() + (self.num_colors * is_rank_hint)

            # Print all values:
            print(f"{hint_action=}")
            print(f"{teammate_idx=}")
            print(f"{card_idx=}")
            print(f"{is_rank_hint=}")
            print(f"{hint_card=}")
            print(f"{hint_card_rank=}")
            print(f"{hint_card_colour=}")
            print(f"{hint_one_hot=}")
            print(f"{hint_value=}")

            hints_per_player = self.num_colors + self.num_ranks
            # First hand_size * 2 actions are discards and plays.
            return self.hand_size * 2 + hints_per_player * teammate_idx + hint_value

        is_discard_play = action < (2 * self.hand_size)
        return lax.cond(
            is_discard_play,
            convert_discard_play_action,
            convert_hint_action,
        )

    @partial(jax.jit, static_argnums=[0, 1])
    def state_to_obs(
        self,
        is_first_state: bool,
        old_state: State,
        new_state: State,
        actions: Dict[str, chex.Array],
    ) -> chex.Array:
        """
        Each card encoding has color, rank, player hand, position in hand, and type.
        Assuming 5 colors, 5 ranks, 5 cards per hand:
        - Color: 0 to 5 (5 is "unknown" for cards in own hand)
        - Rank: 0 to 6 (5 is full stack for fireworks, 6 is "unknown" for cards in own
          hand)
        - Player: 0 to num_agents (a value for each player and a value for fireworks)
        - Position: idx in hand, 0 to 6 (position for cards in hand, plus a value for
          fireworks, 5, and a value for hint actions, 6, which don't correspond to
          exact card indices)
        - Type: {observation, action_play, action_discard, action_hint_self,
          action_hint_teammate_1, action_hint_teammate_2, ...}
        """
        fireworks_tokens = {
            aidx: self.get_fireworks_tokens(new_state)
            for aidx in range(self.num_agents)
        }
        hand_tokens = {
            aidx: self.get_hand_tokens(new_state, aidx)
            for aidx in range(self.num_agents)
        }

        card_tokens = {
            aidx: jnp.concatenate((fireworks_tokens[aidx], hand_tokens[aidx]))
            for aidx in range(self.num_agents)
        }

        if not is_first_state:
            # Convert one-hot player index to scalar:
            last_cur_player = old_state.cur_player_idx.argmax()
            # Extract the action of the last current agent (the one that actually acted):
            actions = jnp.array([actions[i] for i in self._env.agents])
            action = actions[last_cur_player]
            action_tokens = {
                aidx: self.get_action_token(old_state, action, last_cur_player, aidx)
                for aidx in range(self.num_agents)
            }
        else:
            action_tokens = {
                aidx: jnp.zeros((self.token_length,)) for aidx in range(self.num_agents)
            }

        card_tokens = {
            aidx: jnp.vstack((action_tokens[aidx], card_tokens[aidx]))
            for aidx in range(self.num_agents)
        }

        extra_tokens = self.get_extra_tokens(new_state)
        name_to_tokens = {
            name: (card_tokens[aidx], *extra_tokens)
            for aidx, name in enumerate(self.agents)
        }
        if self.flatten_obs:
            name_to_tokens = {
                name: jnp.concatenate([t.ravel() for t in tokens])
                for name, tokens in name_to_tokens.items()
            }
        return name_to_tokens

    @partial(jax.jit, static_argnums=[0])
    def get_action_token(
        self,
        old_state: State,
        action: int,
        last_cur_player: int,
        aidx: int,
    ) -> chex.Array:
        def get_play_discard_card_encoding():
            # Get the true card value from the old_state player hands. This info is now
            # accessible to all players:
            card_idx = action - self.hand_size
            hand = old_state.player_hands[last_cur_player]
            # Card must exist for this to be a valid action:
            card = hand[card_idx]
            return self.card_matrix_to_encoding(card)

        def get_hint_color_or_rank():
            hint_action = action - (2 * self.hand_size)
            num_hints = self.num_colors + self.num_ranks
            hint_value = hint_action % num_hints
            # Hint values 0-4 correspond to colors, 5-9 ranks. The hand representation
            # starts with colors, so we can convert this directly to onehot:
            color_or_rank = nn.one_hot(hint_value, num_classes=num_hints)
            return color_or_rank

        is_hint = action >= 2 * self.hand_size
        color_rank_encoding = lax.cond(
            is_hint,
            get_hint_color_or_rank,
            get_play_discard_card_encoding,
        )
        # Play and discard cards are universally known.
        # TODO: For rank hints we could set the color to unknown and vice versa, but
        # leaving as 0 for now in order to not overload the meaning of unknown as used
        # by cards in hand. Might need to reconsider this.
        color_rank_encoding = self.add_extra_card_encodings(color_rank_encoding)

        relative_player_idx = self.get_relative_player_idx_onehot(last_cur_player, aidx)
        # Append 0 (the extra value indicating fireworks):
        relative_player_idx = jnp.concatenate((relative_player_idx, jnp.zeros(1)))

        # The "position" of hints is the max value of the hand size + 1:
        position = lax.select(is_hint, self.hand_size + 1, action % self.hand_size)
        position = nn.one_hot(position, num_classes=self.hand_size + 2)

        def get_hint_target_relative_player_idx():
            hint_action = action - (self.hand_size * 2)
            num_hints = self.num_colors + self.num_ranks
            relative_target_idx = hint_action // num_hints
            # The first hints correspond to the acting agent's next partner
            # (last_cur_player + 1) and so on:
            absolute_target_idx = (
                last_cur_player + relative_target_idx + 1
            ) % self.num_agents
            # Make the absolute player index relative to aidx (the agent whose obs we're
            # generating):
            target_onehot = self.get_relative_player_idx_onehot(
                absolute_target_idx, aidx
            )
            # There are 3 other action types, so prepend 3 zeros:
            return jnp.concatenate((jnp.zeros(3), target_onehot))

        def get_play_discard_action_type():
            is_play = action < self.hand_size
            # 0 is "observation", 1 is "play", 2 is "discard":
            value = lax.select(is_play, 1, 2)
            return nn.one_hot(value, num_classes=self.num_agents + 3)

        action_type = lax.cond(
            is_hint,
            get_hint_target_relative_player_idx,
            get_play_discard_action_type,
        )

        return jnp.concatenate(
            (
                color_rank_encoding,
                relative_player_idx,
                position,
                action_type,
            )
        )

    @partial(jax.jit, static_argnums=[0])
    def get_fireworks_tokens(self, new_state: State):
        # Convert firework ranks to 1-hot encoding:
        ranks = new_state.fireworks.sum(axis=1)
        # Extra row for full fireworks (rank 5) and unknown. The card here corresponds
        # to the rank of the next card that can be played on the fireworks (as suggested
        # by Ravi):
        ranks = nn.one_hot(ranks, num_classes=self.num_ranks + 2)

        # One extra row for unknown:
        colors = jnp.eye(self.num_colors, self.num_colors + 1)

        # Special value for fireworks:
        player = nn.one_hot(self.num_agents, num_classes=self.num_agents + 1)
        position = nn.one_hot(self.hand_size, num_classes=self.hand_size + 2)
        token_type = nn.one_hot(0, num_classes=self.num_agents + 3)

        # Tile the features that are the same for each firework token, so that they can
        # be concatenated with color and ranks:
        fixed_features = jnp.concatenate((player, position, token_type))
        fixed_features = jnp.tile(fixed_features, (self.num_colors, 1))

        return jnp.concatenate((colors, ranks, fixed_features), axis=1)

    @partial(jax.jit, static_argnums=[0])
    def get_hand_tokens(self, new_state: State, aidx: int):
        # First build ground-truth hand token sequence, then mask own hand:
        hands_one_hot = self.card_matrix_to_encoding(new_state.player_hands)
        # Make order of hands relative to aidx:
        hands_one_hot = jnp.roll(hands_one_hot, -aidx, axis=0)
        card_is_present_mask = hands_one_hot.any(axis=-1).ravel()
        # Mask own cards with unknown ranks or colors:
        color_revealed = new_state.colors_revealed[aidx].sum(axis=1)
        rank_revealed = new_state.ranks_revealed[aidx].sum(axis=1)
        hands_one_hot = hands_one_hot.at[0, :, : self.num_colors].multiply(
            jnp.expand_dims(color_revealed, 1)
        )
        hands_one_hot = hands_one_hot.at[0, :, self.num_colors :].multiply(
            jnp.expand_dims(rank_revealed, 1)
        )
        # "max firework" extra encoding comes before "unknown" rank:
        extra_encodings = jnp.vstack(
            (1 - color_revealed, jnp.zeros_like(rank_revealed), 1 - rank_revealed)
        )
        # Extra encodings for other hands:
        # num_teammate_cards = self.hand_size * (self.num_agents - 1)
        # extra_encodings = jnp.concatenate(
        #     (extra_encodings, jnp.zeros((3, num_teammate_cards)))
        # )
        print(f"{color_revealed=}")
        print(f"{rank_revealed=}")
        print(f"{extra_encodings.T=}")
        print(f"{hands_one_hot=}")
        # hands_one_hot = self.add_extra_card_encodings(hands_one_hot, extra_encodings.T)
        hands_one_hot = jnp.concatenate(
            (
                # Own cards:
                self.add_extra_card_encodings(hands_one_hot[:1], extra_encodings.T),
                # Teammates' cards (always known):
                self.add_extra_card_encodings(hands_one_hot[1:]),
            )
        )
        print(f"{hands_one_hot=}")
        input()

        # Stack all cards as rows:
        hands_one_hot = jnp.vstack(hands_one_hot)

        # TODO: Make these encodings in the constructor as they never change:
        players = jnp.eye(self.num_agents, self.num_agents + 1)
        players = jnp.repeat(players, self.hand_size, axis=0)
        positions = jnp.eye(self.hand_size, self.hand_size + 2)
        positions = jnp.tile(positions, (self.num_agents, 1))
        # Type is observation (0):
        token_type = nn.one_hot(0, num_classes=self.num_agents + 3)
        token_type = jnp.tile(token_type, (self.num_agents * self.hand_size, 1))

        full_hand = jnp.concatenate(
            (hands_one_hot, players, positions, token_type),
            axis=1,
        )

        # Zero out the rows corresponding to missing cards. TODO: This is probably not
        # the best way to do this, should supply a mask but this is a detail that we
        # can implement later.
        full_hand = full_hand * jnp.expand_dims(card_is_present_mask, 1)

        return full_hand

    @partial(jax.jit, static_argnums=[0])
    def get_extra_tokens(self, new_state: State):
        """
        Currently, extra tokens are:
        - Hint tokens remaining
        - Lives remaining
        - Cards left in deck
        """
        # Thermometer encodings:
        info_tokens = new_state.info_tokens
        life_tokens = new_state.life_tokens
        remaining_deck_size = new_state.remaining_deck_size
        # Convert thermometer to one hot:
        extra_tokens = (
            nn.one_hot(info_tokens.sum(), num_classes=info_tokens.size),
            nn.one_hot(life_tokens.sum(), num_classes=life_tokens.size),
            nn.one_hot(remaining_deck_size.sum(), num_classes=remaining_deck_size.size),
        )

        return tuple(jnp.expand_dims(token, 0) for token in extra_tokens)

    @partial(jax.jit, static_argnums=[0])
    def get_relative_player_idx_onehot(
        self,
        player_idx_scalar: int,
        own_idx_scalar: int,
    ):
        absolute_idx = nn.one_hot(player_idx_scalar, num_classes=self.num_agents)
        return jnp.roll(absolute_idx, -own_idx_scalar)

    @partial(jax.jit, static_argnums=[0])
    def card_matrix_to_encoding(self, card_matrix: chex.Array) -> chex.Array:
        """
        card_matrix is the representation used in player_hands a "one-hot matrix" of
        shape (num_ranks, num_colors).
        """
        color = card_matrix.sum(axis=-1)
        rank = card_matrix.sum(axis=-2)
        return jnp.concatenate((rank, color), axis=-1)

    @partial(jax.jit, static_argnums=[0])
    def add_extra_card_encodings(self, card_one_hot, values=0):
        # Add the extra elements for "unknown" colors and ranks, and the extra rank for
        # max fireworks:
        last_idx = self.num_colors + self.num_ranks
        extra_idxs = jnp.array([self.num_colors, last_idx, last_idx])
        return jnp.insert(card_one_hot, extra_idxs, values, axis=-1)


def random_wrapped_rollout(print_shapes_only=False, step_through=True, **env_kwargs):
    from jaxmarl.environments.hanabi.hanabi import HanabiEnv

    env = HanabiEnv(**env_kwargs)
    env = HanabiTokenObsWrapper(env, flatten_obs=False)

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    dones = {a: False for a in env.agents}

    print(f"{state=}")
    print(f"{obs=}")
    for action in range(2 * env.hand_size + env.num_colors + env.num_ranks):
        a1 = env.action_to_env_action(state, 0, action)
        a2 = env.action_to_env_action(state, 1, action)
        print(f"{obs['agent_0'][0][env.num_colors + 1 + action // 2]=}")
        print(f"{obs['agent_1'][0][env.num_colors + 1 + action // 2]=}")
        print(f"{action=} {a1=} {a2=}")

    while True:
        if step_through:
            input("Press enter to step...")

        key, _key = jax.random.split(key)
        num_hints = env.num_colors + env.num_ranks
        actions = jax.random.randint(
            _key,
            [env.num_agents],
            0,
            2 * env.hand_size + num_hints * (env.num_agents - 1),
        )
        actions = {a: action for a, action in zip(env.agents, actions)}
        # pprint(actions)
        key, _key = jax.random.split(key)
        pprint(state)
        obs, state, rewards, dones, infos = env.step(_key, state, actions)
        # pprint(obs)


if __name__ == "__main__":
    jax.config.update("jax_disable_jit", True)
    random_wrapped_rollout()
