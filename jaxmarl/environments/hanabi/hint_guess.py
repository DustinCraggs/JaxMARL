import jax
import jax.numpy as jnp
import numpy as np
import chex
import copy

from flax import struct
from gymnax.environments.spaces import Discrete, Box
from functools import partial
from itertools import product

from jaxmarl.environments.multi_agent_env import MultiAgentEnv


@struct.dataclass
class State:
    player_hands: chex.Array
    target: chex.Array
    hint: chex.Array
    guess: chex.Array
    turn: int


class HintGuessGame(MultiAgentEnv):

    def __init__(
        self,
        num_agents=2,
        num_features=2,
        num_classes_per_feature=[3, 3],
        hand_size=5,
        card_encoding="onehot",
        matrix_obs=False,
        card_idx_actions=False,
        agents=None,
        action_spaces=None,
        observation_spaces=None,
    ):
        super().__init__(num_agents)

        assert num_agents == 2, "Environment defined only for 2 agents"

        if agents is None:
            self.agents = ["hinter", "guesser"]
        self.num_agents = num_agents
        self.agent_range = jnp.arange(num_agents)

        self.hand_size = hand_size
        self.num_features = num_features
        self.num_classes_per_feature = num_classes_per_feature
        self.num_cards = np.prod(self.num_classes_per_feature)
        self.matrix_obs = matrix_obs
        self.card_idx_actions = card_idx_actions
        self.card_feature_space = jnp.array(
            list(product(*[np.arange(n_c) for n_c in self.num_classes_per_feature]))
        )

        # generate the deck of one-hot encoded cards
        if card_encoding == "onehot":
            self.encoding_dim = np.sum(self.num_classes_per_feature)
            self.card_encodings = self.get_onehot_encodings()
        else:
            raise NotImplementedError("Available encodings are: 'onehot'")

        # The transformer agent needs explicit info indicating:
        #   - Whose hand each card belongs to (or special card)
        #   - If it is the hinter or the guesser
        #   - If it is currently the first or second turn
        # Pre-compute observation encodings:
        # Card ownership encodings (own cards, partner's cards, or special card):
        num_card_owners = self.num_agents + 1
        card_owner_encodings = jnp.eye(num_card_owners)
        agent_ids = jnp.repeat(jnp.arange(num_agents), hand_size)
        # Shaped for concatenation later:
        self.agent_hand_encodings = card_owner_encodings[agent_ids]
        self.special_card_encoding = card_owner_encodings[-1]
        # Agent role (can be hinter or guesser):
        num_roles = 2
        self.role_encodings = jnp.eye(num_roles)
        # Turn type (first or second turn):
        num_turns = 2
        self.turn_encodings = jnp.eye(num_turns)

        # Expand the encoding dim to accomodate above extra encodings:
        num_obs_cards = self.num_agents * (self.hand_size) + 1
        encoded_card_size = self.encoding_dim + num_card_owners + num_roles + num_turns
        self.obs_size = num_obs_cards * encoded_card_size

        if action_spaces is None:
            if self.card_idx_actions:
                self.action_dim = self.hand_size + 1
            else:
                # hint-guess one card of the game + nothing
                self.action_dim = np.prod(self.num_classes_per_feature) + 1
            self.action_spaces = {i: Discrete(self.action_dim) for i in self.agents}
        if observation_spaces is None:
            if self.matrix_obs:
                obs_space = Box(low=0, high=1, shape=(num_obs_cards, encoded_card_size))
                self.observation_spaces = {i: obs_space for i in self.agents}
            else:
                # TODO: This space is technically not correct, should be MultiDiscrete:
                self.observation_spaces = {
                    i: Discrete(self.obs_size) for i in self.agents
                }

    @partial(jax.jit, static_argnums=[0])
    def reset(self, rng):
        rng_hands, rng_target = jax.random.split(rng)
        player_hands = jax.random.choice(
            rng_hands,
            jnp.arange(self.num_cards),
            shape=(
                self.num_agents,
                self.hand_size,
            ),
        )

        # Duplicate the hands (one set for each agent):
        player_hands = jnp.tile(player_hands, (2, 1, 1))
        # Only shuffle one hand in one agent's state. This makes the index
        # correspondence between the two hands different in each agent's state, which
        # should make index-based policies impossible:
        rng_hands, rng = jax.random.split(rng_hands)
        player_hands = player_hands.at[0, 0, :].set(
            jax.random.permutation(rng, player_hands[0, 0, :])
        )

        # The target is a random card from the second agent's hand:
        target = jax.random.choice(rng_target, player_hands[1, 1, :])

        state = State(
            player_hands=player_hands, target=target, hint=-1, guess=-1, turn=0
        )
        return jax.lax.stop_gradient(self.get_obs(state)), state

    @partial(jax.jit, static_argnums=[0, 2, 3])
    def reset_for_eval(self, rng, reset_mode="exact_match", replace=True):

        def get_safe_probability_from_mask(mask):
            def true_fn(mask):
                return jnp.full(self.num_cards, -1, dtype=jnp.float32)

            return jax.lax.cond(
                jnp.sum(mask) == 0, true_fn, lambda x: x / x.sum(), mask
            )

        def mask_selected_card(args):
            hand_masks, selected_cards = args
            masked_mask = jax.vmap(lambda mask, card: mask.at[card].set(0))(
                hand_masks, selected_cards
            )
            return masked_mask

        def get_random_pair(masks, rng, replace=False):
            """
            This function returns a random pair of cards based on masks such that there
            is no relation between the cards.
            """
            h_rng, g_rng = jax.random.split(rng, 2)
            task_specific_mask, h_hand_mask, g_hand_mask = masks
            h_hand_mask = task_specific_mask * h_hand_mask
            h_card = jax.random.choice(
                h_rng, self.num_cards, p=get_safe_probability_from_mask(h_hand_mask)
            )
            g_hand_mask = task_specific_mask * g_hand_mask
            g_card = jax.random.choice(
                g_rng, self.num_cards, p=get_safe_probability_from_mask(g_hand_mask)
            )
            selected_cards = jnp.array([h_card, g_card])
            hand_masks = jnp.stack([h_hand_mask, g_hand_mask], axis=0)
            updated_masks = jax.lax.cond(
                replace,
                mask_selected_card,
                lambda args: args[0],
                (hand_masks, selected_cards),
            )
            return updated_masks, selected_cards

        def get_identical_pair(masks, rng, replace=False):
            """
            This function returns a pair of cards that are identical in all the
            features.
            """
            task_specific_mask, h_hand_mask, g_hand_mask = masks
            both_hand_available = h_hand_mask * g_hand_mask
            card_mask = task_specific_mask * both_hand_available
            card_drawn = jax.random.choice(
                rng,
                jnp.arange(self.num_cards),
                p=get_safe_probability_from_mask(card_mask),
            )
            selected_cards = jnp.array([card_drawn, card_drawn])
            hand_masks = jnp.stack([h_hand_mask, g_hand_mask], axis=0)
            updated_masks = jax.lax.cond(
                replace,
                mask_selected_card,
                lambda args: args[0],
                (hand_masks, selected_cards),
            )
            return updated_masks, selected_cards

        def get_similar_pair(masks, rng, replace=False):
            """
            This function returns a pair of cards that are identical in at least one
            feature, could be identical
            """
            f_rng, h_rng, g_rng = jax.random.split(rng, 3)
            task_specific_mask, h_hand_mask, g_hand_mask = masks
            h_card_mask = task_specific_mask * h_hand_mask
            # Robustness issue: need to solve the problem where there might not be a
            # possible corresponding card for the other player
            # Solution: need to backtrack a set of possible hints from guesser's
            # playable cards

            h_card = jax.random.choice(
                h_rng, self.num_cards, p=get_safe_probability_from_mask(h_card_mask)
            )
            h_card_similarity_mask = jnp.any(
                get_similarity_mask(h_card), axis=0
            ).astype(jnp.int32)
            # accept identcal cards as a dirty fix for now
            non_h_cards = 1 - jax.nn.one_hot(h_card, self.num_cards)
            g_card_mask = (
                task_specific_mask * g_hand_mask * h_card_similarity_mask * non_h_cards
            )
            g_card = jax.random.choice(
                g_rng, self.num_cards, p=get_safe_probability_from_mask(g_card_mask)
            )
            selected_cards = jnp.array([h_card, g_card])
            hand_masks = jnp.stack([h_hand_mask, g_hand_mask], axis=0)
            updated_masks = jax.lax.cond(
                replace,
                mask_selected_card,
                lambda args: args[0],
                (hand_masks, selected_cards),
            )
            return updated_masks, selected_cards

        def get_non_simillar_pair(masks, rng, replace=False):
            """
            This function returns a pair of cards that are different in all features.
            """

            def get_non_sim_feature_mask(card):
                def feature_level_non_similar_mask(label, arr):
                    return jnp.where(label != arr, 1, 0)

                card_feature_vector = self.card_feature_space[card, :]
                card_non_similar_masks = jax.vmap(
                    feature_level_non_similar_mask, in_axes=(0, 1)
                )(card_feature_vector, self.card_feature_space)
                return jnp.prod(card_non_similar_masks, axis=0)

            h_rng, g_rng = jax.random.split(rng, 2)
            task_specific_mask, h_hand_mask, g_hand_mask = masks
            h_card_mask = task_specific_mask * h_hand_mask
            h_card = jax.random.choice(
                h_rng, self.num_cards, p=get_safe_probability_from_mask(h_card_mask)
            )
            h_non_sim_feature_mask = get_non_sim_feature_mask(h_card)
            g_card_mask = task_specific_mask * g_hand_mask * h_non_sim_feature_mask
            g_card = jax.random.choice(
                g_rng, self.num_cards, p=get_safe_probability_from_mask(g_card_mask)
            )
            selected_cards = jnp.array([h_card, g_card])
            hand_masks = jnp.stack([h_hand_mask, g_hand_mask], axis=0)
            updated_masks = jax.lax.cond(
                replace,
                mask_selected_card,
                lambda args: args[0],
                (hand_masks, selected_cards),
            )
            return updated_masks, selected_cards

        def get_similarity_mask(card):
            def feature_level_similar_mask(label, arr):
                return jnp.where(label == arr, 1, 0)

            # card_similar_masks is an ndarray indicating the which cards are similar on
            # which features of the target it has shape feature x num_cards
            card_feature_vector = self.card_feature_space[card, :]
            card_similar_masks = jax.vmap(feature_level_similar_mask, in_axes=(0, 1))(
                card_feature_vector, self.card_feature_space
            )
            return card_similar_masks

        feature_masks = jnp.ones((self.num_features, self.num_cards), dtype=jnp.int32)
        h_hand_mask = jnp.ones(self.num_cards, dtype=jnp.int32)
        g_hand_mask = jnp.ones(self.num_cards, dtype=jnp.int32)
        _, rng, rng_hand, rng_view = jax.random.split(rng, 4)

        # when generating hands, the hint/target pair are first generated, then the rest
        # of the hands are generated
        if reset_mode == "exact_match":
            hint_target_fn = get_identical_pair
            rest_of_hands_fn = get_random_pair
        elif reset_mode == "similarity_match":
            hint_target_fn = get_similar_pair
            rest_of_hands_fn = get_random_pair
        elif reset_mode == "mutual_exclusive":
            hint_target_fn = get_non_simillar_pair
            rest_of_hands_fn = get_identical_pair
        elif reset_mode == "mutual_exclusive_similarity":
            hint_target_fn = get_non_simillar_pair
            rest_of_hands_fn = get_similar_pair
        else:
            raise ValueError("Invalid reset mode")

        # during the target/hint generation phase, there is no restictions on what
        # target can be, thus feature masks are all 1 replace is forced to be true to
        # guarentee that hint/target are only used once from individual set
        hand_masks, selected_cards = hint_target_fn(
            (jnp.ones(self.num_cards, dtype=jnp.int32), h_hand_mask, g_hand_mask),
            rng,
            replace=True,
        )
        h_hand_mask, g_hand_mask = hand_masks[0, :], hand_masks[1, :]

        # set the target card in hinter's hand to 0 to avoid duplication if they are not
        # the same
        h_hand_mask = h_hand_mask * g_hand_mask

        # similarly, set the hint card in guesser's hand to 0 to avoid duplication if
        # they are not the same, e.g., if hint is 6, then 6 must not be an option for
        # guesser's rest of the hand
        g_hand_mask = g_hand_mask * h_hand_mask

        hint, target = selected_cards[0], selected_cards[1]
        similarity_mask = get_similarity_mask(target)

        if reset_mode == "exact_match":
            # there is no other restriction on values of rest of the hands, apart from
            # hand masks on hint/target
            task_specific_mask = jnp.ones(self.num_cards, dtype=jnp.int32)
        elif reset_mode == "similarity_match":
            # apart from hint/target, the rest of the hands pairs should also have no
            # feature that is same as target
            task_specific_mask = 1 - jnp.any(similarity_mask, axis=0).astype(jnp.int32)
        elif reset_mode == "mutual_exclusive":
            # apart from hint/target, the rest of the hands pairs should also have no
            # feature that is same as target
            task_specific_mask = 1 - jnp.any(similarity_mask, axis=0).astype(jnp.int32)
        else:  # the mutual_exclusive_similarity case
            # set the rest of the hand to be "no feature idential", like the two
            # previous cases
            task_specific_mask = 1 - jnp.any(similarity_mask, axis=0).astype(jnp.int32)
            # future case:  we can let the rest of the hands to have one/more similar
            #               feature with the target, as far as they are not the target
            #               but this involves a more complex logic to ensure that the
            #               rest of hand does not hint other card of the rest of the
            #               hand a potential solution is to scan the current hand
            #               generated and backtrack if the rest of hand does not
            #               implicitly hint each other

        hinter_hand = hint
        guesser_hand = target
        for _ in range(self.hand_size - 1):
            _, rng = jax.random.split(rng)
            hand_masks, selected_cards = rest_of_hands_fn(
                (task_specific_mask, h_hand_mask, g_hand_mask), rng, replace=replace
            )
            h_hand_mask, g_hand_mask = hand_masks[0, :], hand_masks[1, :]
            hinter_card, guesser_card = selected_cards[0], selected_cards[1]
            hinter_hand = jnp.append(hinter_hand, hinter_card)
            guesser_hand = jnp.append(guesser_hand, guesser_card)

        # shuffle the cards
        _rngs = jax.random.split(rng_hand, 2)
        hinter_hand = jax.random.permutation(_rngs[0], hinter_hand)
        guesser_hand = jax.random.permutation(_rngs[1], guesser_hand)

        player_hands = jnp.stack([hinter_hand, guesser_hand])
        # assert player_hands.shape == (2, self.hand_size)

        # shuffle the views
        _rngs = jax.random.split(rng_view, self.num_agents)
        permuted_hands = jax.vmap(
            lambda rng: jax.random.permutation(rng, player_hands, axis=1)
        )(_rngs)
        state = State(
            player_hands=permuted_hands, target=target, hint=-1, guess=-1, turn=0
        )

        return jax.lax.stop_gradient(self.get_obs(state)), state, hint

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, rng, state, actions):

        def step_hint(state, actions):
            action = actions["hinter"]

            if self.card_idx_actions:
                # Hinter's obs is index 0, and their hand is at index 0. Subtract 1
                # from the action to account for no-op action (0):
                action = state.player_hands[0, 0, action - 1]

            state = state.replace(
                hint=action,
                turn=1,
            )
            reward = 0
            done = False
            return state, reward, done

        def step_guess(state, actions):
            action = actions["guesser"]

            if self.card_idx_actions:
                # Guesser's obs and hand are both at index 1. Subtract 1 from the action
                # to account for no-op action (0):
                action = state.player_hands[1, 1, action - 1]

            state = state.replace(
                guess=action,
                turn=2,
            )
            reward = (action == state.target).astype(int)
            done = True
            return state, reward, done

        state, reward, done = jax.lax.cond(
            state.turn == 0, step_hint, step_guess, state, actions
        )

        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        rewards = {agent: reward for agent in self.agents}
        info = {}

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            rewards,
            dones,
            info,
        )

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state):
        """
        Obs is [one_hot(feat1),one_hot(feat2)...,agent_id_hand,agent_id_obs] per
        each card in all the hands.
        """

        target_hint_card = jnp.where(
            state.turn == 0,  # is hint step?
            self.card_encodings[state.target],  # target if is hint step
            self.card_encodings[state.hint],  # hint otherwise
        )
        # Add the target card encoding:
        target_hint_card = jnp.append(target_hint_card, self.special_card_encoding)

        def _get_obs(aidx):
            # Build the hands:
            hands = state.player_hands[aidx]
            card_encodings = self.card_encodings[hands]

            # Move this agent's hand to the first position (i.e. for the guesser's obs,
            # move its hand to the first position):
            card_encodings = jnp.roll(card_encodings, shift=aidx, axis=0)

            # Stack both agent's hands as rows:
            card_encodings = jnp.vstack(card_encodings)

            # Add the hand encodings (own card or partner's card):
            card_encodings = jnp.concatenate(
                (card_encodings, self.agent_hand_encodings), axis=-1
            )

            # Mask the target card for the guesser at turn 0:
            target_hint_card_masked = jnp.where(
                (aidx == 1) & (state.turn == 0),
                jnp.zeros_like(target_hint_card),
                target_hint_card,
            )
            card_encodings = jnp.concatenate(
                (card_encodings, target_hint_card_masked[jnp.newaxis])
            )

            # Add the role and turn encodings (this could just be a separate token in
            # future, which might be better):
            role_and_turn_encs = jnp.concatenate(
                (self.role_encodings[aidx], self.turn_encodings[state.turn])
            )
            num_card_encs = card_encodings.shape[0]
            role_and_turn_encs = jnp.tile(role_and_turn_encs, (num_card_encs, 1))
            card_encodings = jnp.concatenate(
                (card_encodings, role_and_turn_encs), axis=-1
            )

            # Flatten if the obs is not requested as matrix:
            if not self.matrix_obs:
                card_encodings = card_encodings.ravel()

            return card_encodings

        obs = jax.vmap(_get_obs)(jnp.arange(self.num_agents))

        return {"hinter": obs[0], "guesser": obs[1]}

    @partial(jax.jit, static_argnums=[0])
    def get_legal_moves(self, state):
        """
        Legal moves in first step are the features-combinations cards for the hinter,
        nope for the guesser. Symmetric for second round
        """

        if self.card_idx_actions:
            # Noop move is the first action:
            # TODO: This is a hack to work with current model impl.
            noop_move = jnp.zeros(self.action_dim).at[0].set(1)
            all_card_actions = jnp.ones(self.action_dim).at[0].set(0)
            # Legal moves for turn 0:
            legal_moves = jnp.stack((all_card_actions, noop_move))
            # Rotate if it's turn two:
            legal_moves = jnp.roll(legal_moves, shift=state.turn, axis=0)
            return {"hinter": legal_moves[0], "guesser": legal_moves[1]}

        # Only last action ("do nothing") is valid:
        noop_move = jnp.zeros(self.action_dim).at[-1].set(1)
        actions = jnp.zeros(self.action_dim)

        # For hint turn, available actions are the card types of the hinter:
        hinter_legal_moves = jnp.where(
            state.turn == 0, actions.at[state.player_hands[0][0]].set(1), noop_move
        )

        # For guess turn, available actions are the card types of the guesser:
        guesser_legal_moves = jnp.where(
            state.turn == 1, actions.at[state.player_hands[0][1]].set(1), noop_move
        )

        return {"hinter": hinter_legal_moves, "guesser": guesser_legal_moves}

    def get_onehot_encodings(self):
        """
        Concatenation of one_hots for every card feature, f.i., 2feats with
        2classes -> [1,0,0,1,0,0,0]
        """
        encodings = [
            jax.nn.one_hot(jnp.arange(n_c), n_c) for n_c in self.num_classes_per_feature
        ]
        encodings = jnp.array(
            [jnp.concatenate(combination) for combination in list(product(*encodings))]
        )
        return encodings


if __name__ == "__main__":
    # jax.config.update("jax_disable_jit", True)

    from jaxmarl import make

    env = HintGuessGame()

    for i in range(1000):
        rng = jax.random.PRNGKey(i)

        # reset_modes: exact_match, similarity_match, mutual_exclusive,
        # mutual_exclusive_similarity
        obs, state, correct_hint = env.reset_for_eval(
            rng, reset_mode="mutual_exclusive_similarity", replace=False
        )

        rng, _rng = jax.random.split(rng)
        key_act = jax.random.split(_rng, env.num_agents)
        hint_actions = {
            agent: env.action_space(agent).sample(key_act[i])
            for i, agent in enumerate(env.agents)
        }

        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, infos = env.step(_rng, state, hint_actions)

        rng, _rng = jax.random.split(rng)
        key_act = jax.random.split(_rng, env.num_agents)
        guess_actions = {
            agent: env.action_space(agent).sample(key_act[i])
            for i, agent in enumerate(env.agents)
        }

        obs, _state, reward, done, infos = env.step(_rng, state, guess_actions)

        if reward["guesser"] != 0:

            print(
                "hint: ",
                hint_actions["hinter"],
                ", correct hint: ",
                correct_hint,
                ", guess: ",
                guess_actions["guesser"],
                ", correct guess: ",
                state.target,
                ", reward: ",
                reward,
                ", score: ",
                (hint_actions["hinter"] == correct_hint)
                & (
                    guess_actions["guesser"] == state.target
                ),  # correct hint and correct guess
            )
