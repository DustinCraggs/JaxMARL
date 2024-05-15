import jax
import jax.numpy as jnp


def compute_hinter_metrics(
    player_hands,
    target_card,
    action,
    card_idx_to_features,
    card_idx_actions,
):
    hinter_hand = player_hands[0, 0]
    guesser_hand = player_hands[0, 1]

    # 0 is the no-op action, so the card index is (action - 1):
    action_card = hinter_hand[action - 1] if card_idx_actions else action

    # Convert card indices to feature vectors:
    hinter_hand = card_idx_to_features[hinter_hand]
    guesser_hand = card_idx_to_features[guesser_hand]
    target_card = card_idx_to_features[target_card]
    action_card = card_idx_to_features[action_card]

    # Get feature matches for both hands:
    hinter_hand_feature_matches = jnp.equal(hinter_hand, target_card)
    hinter_hand_num_matches = hinter_hand_feature_matches.sum(axis=1)
    guesser_hand_feature_matches = jnp.equal(guesser_hand, target_card)
    guesser_hand_num_matches = guesser_hand_feature_matches.sum(axis=1)

    num_similarity_ambiguities = get_num_similarity_ambiguities(
        hinter_hand_feature_matches,
        hinter_hand_num_matches,
        guesser_hand_feature_matches,
        guesser_hand_num_matches,
    )

    num_dissimilarity_ambiguities = get_num_dissimilarity_ambiguities(
        hinter_hand_feature_matches,
        hinter_hand_num_matches,
        guesser_hand_feature_matches,
        guesser_hand_num_matches,
    )

    # For future experiments where we may use ordinal (e.g. sinusoidal) encodings,
    # enabling policies to learn a "distance" metric for card features:
    hinter_hand_distances = jnp.abs(hinter_hand - target_card).sum(axis=1)
    action_distance = jnp.abs(action_card - target_card).sum()

    num_feature_matches = (action_card == target_card).sum()
    max_possible_feature_matches = hinter_hand_num_matches.max()
    min_possible_feature_matches = hinter_hand_num_matches.min()

    is_highest_feature_match = num_feature_matches == max_possible_feature_matches
    is_lowest_feature_match = num_feature_matches == min_possible_feature_matches

    metrics = {
        # Similarity strategies:
        "exact_match": (action_card == target_card).all(),
        "exact_match_possible": hinter_hand_distances.min() == 0,
        "feature_sim_hint": is_highest_feature_match,
        "unambiguous_sim_hint_possible": num_similarity_ambiguities == 0,
        # Dissimilarity strategies:
        "feature_dissim_hint": is_lowest_feature_match,
        "unambiguous_dissim_hint_possible": num_dissimilarity_ambiguities == 0,
        # Distance-based metrics:
        "action_distance": action_distance,
        "max_guess_distance": hinter_hand_distances.max(),
        "min_guess_distance": hinter_hand_distances.min(),
    }

    return metrics


# Compute number of feature matches in the same position for all pairs of cards
# in masked hands. Keep the max across the hinter's cards:
def get_num_similarity_ambiguities(
    hinter_hand_feature_matches,
    hinter_hand_num_matches,
    guesser_hand_feature_matches,
    guesser_hand_num_matches,
):
    """
    Check whether the most and least similar hinter cards lead to ambiguity when using
    a similarity or dissimilarity policy (i.e. is there an equally similar/dissimilar
    card in the guesser's hand to the closest possible hint).
    """
    # Compute number of matching features across all pairs of one hinter and one guesser
    # card:
    num_matches = hinter_hand_feature_matches @ guesser_hand_feature_matches.T
    num_matches = num_matches.astype(int)

    # Mask columns corresponding to the target card in the guesser's hand:
    not_target_mask = guesser_hand_num_matches != guesser_hand_num_matches.max()
    num_matches = num_matches * not_target_mask
    # Take the max number of similar features for each hinter card:
    num_matches = num_matches.max(axis=0)

    # For similarity, the hint is always the card that is most similar to target:
    similarity_hint_matches = hinter_hand_num_matches.max()
    # A similarity hint card is ambiguous if any non-target guesser cards have the
    # same number or more features matching the "best" hint as the target card:
    num_ambiguities = (num_matches >= similarity_hint_matches).sum()
    return num_ambiguities


def get_num_dissimilarity_ambiguities(
    hinter_hand_feature_matches,
    hinter_hand_num_matches,
    guesser_hand_feature_matches,
    guesser_hand_num_matches,
):
    """
    Same as get_num_similarity_ambiguities, but for dissimilarity strategies.
    """
    num_matches = hinter_hand_feature_matches @ guesser_hand_feature_matches.T
    num_matches = num_matches.astype(int)
    # Set targets to max value to ensure they are not selected as the most dissimilar:
    not_target_mask = guesser_hand_num_matches != guesser_hand_num_matches.max()
    num_matches = jnp.where(not_target_mask, num_matches, num_matches.max())
    num_matches = num_matches.min(axis=0)

    dissimilarity_hint_matches = hinter_hand_num_matches.min()
    num_ambiguities = (num_matches <= dissimilarity_hint_matches).sum()
    return num_ambiguities
