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
    # Things that are missing from this obs in the absence of AOH:
    # - Teammate hinted info
    # - Negative hint info
    # - Discarded cards

    def __init__(
        self,
        env: MultiAgentEnv,
        use_token_actions=True,
        add_info_tokens=True,
        add_life_tokens=True,
        add_remaining_deck_size=True,
        add_v0_belief=False,
        add_original_obs=False,
        flatten_obs=False,
    ):
        super().__init__(env)
        self.use_token_actions = use_token_actions
        self.add_info_tokens = add_info_tokens
        self.add_life_tokens = add_life_tokens
        self.add_remaining_deck_size = add_remaining_deck_size
        self.add_v0_belief = add_v0_belief
        self.add_original_obs = add_original_obs
        self.flatten_obs = flatten_obs

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
        obs = self.state_to_obs(True, None, state, None, obs)
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
            # state_to_obs method expects env actions, so convert first:
            action = self.process_actions(state, action)

        # TODO: how much performance improvement would there be if we skip creation
        # of the default observations?
        old_state = state
        obs, new_state, reward, done, info = self._env.step(key, old_state, action)

        obs = self.state_to_obs(False, old_state, new_state, action, obs)
        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=[0])
    def process_actions(self, state, actions):
        actions = {
            name: self.action_to_env_action(state, aidx, actions[name])
            for aidx, name in enumerate(self.agents)
        }
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
        obs: chex.Array,
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
        fireworks_tokens = self.get_fireworks_tokens(new_state)

        hand_tokens = {
            aidx: self.get_hand_tokens(new_state, aidx)
            for aidx in range(self.num_agents)
        }

        action_tokens = self.get_aidx_to_action_tokens(
            is_first_state, old_state, actions
        )

        card_tokens = {
            aidx: jnp.concatenate(
                (action_tokens[aidx], hand_tokens[aidx], fireworks_tokens)
            )
            for aidx in range(self.num_agents)
        }

        # Add extra tokens:
        extra_tokens = self.get_extra_tokens(new_state)
        name_to_tokens = {
            name: (
                card_tokens[aidx],
                *self.get_extra_tokens_for_agent(new_state, aidx),
                *extra_tokens,
            )
            for aidx, name in enumerate(self.agents)
        }

        if self.add_original_obs:
            name_to_tokens = {
                name: (*name_to_tokens[name], jnp.expand_dims(obs[name], 0))
                for name in self.agents
            }

        if self.flatten_obs:
            name_to_tokens = {
                name: jnp.concatenate([t.ravel() for t in tokens])
                for name, tokens in name_to_tokens.items()
            }
        return name_to_tokens

    @partial(jax.jit, static_argnums=[0, 1])
    def get_aidx_to_action_tokens(self, is_first_state, old_state, actions):
        if is_first_state:
            # Add a null action for the first observation (means obs are always
            # the same size, but ultimately this should ideally be masked):
            null_action = jnp.zeros((1, self.token_length))
            return {aidx: null_action for aidx in range(self.num_agents)}
        # Convert one-hot player index to scalar:
        last_cur_player = old_state.cur_player_idx.argmax()
        # Extract the action of the last current agent (the one that actually acted):
        actions = jnp.array([actions[a] for a in self.agents])
        action = actions[last_cur_player]
        action_tokens = {
            aidx: self.get_action_token(old_state, action, last_cur_player, aidx)
            for aidx in range(self.num_agents)
        }
        return {aidx: jnp.expand_dims(tok, 0) for aidx, tok in action_tokens.items()}

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

        # The "position" of hints is the max value of hand_size + 1:
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
            is_discard = action < self.hand_size
            # value = lax.select(is_discard, 1, 2)
            # 0 is "observation", 1 is "play", 2 is "discard". num_agents possible hint
            # targets, plus observation, discard, and play:
            return nn.one_hot(1 + is_discard, num_classes=self.num_agents + 3)

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
        # One extra row for unknown:
        colors = jnp.eye(self.num_colors, self.num_colors + 1)

        # Convert firework ranks to 1-hot encoding:
        ranks = new_state.fireworks.sum(axis=1)
        # Extra col for full fireworks (rank 5) and unknown. The card here corresponds
        # to the rank of the next card that can be played on the fireworks (as suggested
        # by Ravi):
        ranks = nn.one_hot(ranks, num_classes=self.num_ranks + 2)

        # Special values for fireworks:
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
        # Add extra encodings for own hand, and zeros for teammates' hands:
        hands_one_hot = jnp.concatenate(
            (
                # Own cards:
                self.add_extra_card_encodings(hands_one_hot[:1], extra_encodings.T),
                # Teammates' cards (always known):
                self.add_extra_card_encodings(hands_one_hot[1:]),
            )
        )

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
        extra_tokens = []

        # Thermometer encodings:
        if self.add_info_tokens:
            info_tokens = new_state.info_tokens
            extra_tokens.append(
                nn.one_hot(info_tokens.sum(), num_classes=info_tokens.size + 1)
            )
        if self.add_life_tokens:
            life_tokens = new_state.life_tokens
            extra_tokens.append(
                nn.one_hot(life_tokens.sum(), num_classes=life_tokens.size + 1)
            )
        if self.add_remaining_deck_size:
            remaining_deck_size = new_state.remaining_deck_size
            num_remaining = remaining_deck_size.sum()
            extra_tokens.append(
                nn.one_hot(num_remaining, num_classes=remaining_deck_size.size + 1)
            )
        if self.add_v0_belief:
            belief_v0 = self.get_v0_belief_own_hand_tokens(new_state, 0)
        return tuple(jnp.expand_dims(token, 0) for token in extra_tokens)

    @partial(jax.jit, static_argnums=[0])
    def get_extra_tokens_for_agent(self, new_state: State, aidx: int):
        extra_tokens = []
        if self.add_v0_belief:
            belief_v0 = self.get_v0_belief_own_hand_tokens(new_state, aidx).ravel()
            extra_tokens.append(belief_v0)
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
        card_matrix is the representation used in player_hands, a one-hot matrix of
        shape (..., num_colors, num_ranks).
        """
        color = card_matrix.sum(axis=-1)
        rank = card_matrix.sum(axis=-2)
        # return jnp.concatenate((rank, color), axis=-1)
        return jnp.concatenate((color, rank), axis=-1)

    @partial(jax.jit, static_argnums=[0])
    def add_extra_card_encodings(self, card_one_hot, values=0):
        # Add the extra elements for "unknown" colors and ranks, and the extra rank for
        # max fireworks:
        last_idx = self.num_colors + self.num_ranks

        # SWAP COLOURS AND RANKS and check for assumptions elsewhere
        extra_idxs = jnp.array([self.num_colors, last_idx, last_idx])
        return jnp.insert(card_one_hot, extra_idxs, values, axis=-1)

    @partial(jax.jit, static_argnums=[0])
    def get_v0_belief_own_hand_tokens(self, state: State, aidx: int):
        """
        Adapted from hanabi.py get_v0_belief_feats.
        TODO: Why does the count not incorporate knowledge of cards in other players'
        hands?
        """
        full_deck = self.get_full_deck()
        remaining_card_counts = (
            full_deck.sum(axis=0).ravel()
            - state.discard_pile.sum(axis=0).ravel()
            - state.fireworks.ravel()
        )
        # Remove counts for cards that have been precluded by hint info:
        belief_v0 = state.card_knowledge[aidx] * remaining_card_counts
        # Normalize:
        belief_v0 /= belief_v0.sum(axis=1)[:, jnp.newaxis]
        return belief_v0.reshape((self.hand_size, self.num_colors, self.num_ranks))

    def render_token_obs(self, obs):
        lines = []
        for agent, agent_obs in obs.items():
            lines.append(f"Obs for {agent}")
            lines.append("\tHand:")
            # Add card representations:
            lines.extend(f"\t\t{self.render_card_token(card)}" for card in agent_obs[0])

            agent_obs_idx = 1
            if self.add_info_tokens:
                lines.append(f"\tInfo tokens: {agent_obs[agent_obs_idx][0]}")
                agent_obs_idx += 1
            if self.add_life_tokens:
                lines.append(f"\tLife tokens: {agent_obs[agent_obs_idx][0]}")
                agent_obs_idx += 1
            if self.add_remaining_deck_size:
                one_line_str = "".join(str(agent_obs[agent_obs_idx][0]).split("\n"))
                lines.append(f"\tRemaining deck size: {one_line_str}")
                agent_obs_idx += 1
        return "\n".join(lines)

    def render_card_token(self, card_token_obs):
        card_schema = [
            ("Color", self.num_colors, " "),
            ("(Unknown?", 1, ") "),
            ("Rank", self.num_ranks, " "),
            ("(Max Firework?", 1, ", "),
            ("Unknown?", 1, ") "),
            ("Player", self.num_agents, " "),
            ("(Firework?", 1, ") "),
            ("Position", self.hand_size, " "),
            ("(Firework?", 1, ", "),
            ("Hint?", 1, ") "),
            ("Type", self.num_agents + 3, ""),
        ]

        expected_length = sum(length for _, length, _ in card_schema)
        if len(card_token_obs) != expected_length:
            raise ValueError(
                f"Expected {expected_length} values in card token, got "
                f"{len(card_token_obs)} for card token {card_token_obs}"
            )

        strings = []
        for name, length, suffix in card_schema:
            strings.append(f"{name}: {card_token_obs[:length]}{suffix}")
            card_token_obs = card_token_obs[length:]

        return "".join(strings)


def random_wrapped_rollout(print_shapes_only=False, step_through=True, **env_kwargs):
    from jaxmarl.environments.hanabi.hanabi import HanabiEnv

    env = HanabiEnv(**env_kwargs)
    env = HanabiTokenObsWrapper(env, flatten_obs=False)

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    print(env.render_token_obs(obs))

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
        pprint(actions)
        key, _key = jax.random.split(key)
        # pprint(state)
        obs, state, rewards, dones, infos = env.step(_key, state, actions)
        env.get_v0_belief_own_hand_tokens(state, 0)
        print(env.render_token_obs(obs))


if __name__ == "__main__":
    jax.config.update("jax_disable_jit", True)
    random_wrapped_rollout()
