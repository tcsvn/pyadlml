from analysis import training
from parsing import digit
from config import settings

import hmmlearn.hmm as hmm
import numpy as np

def get_samplings(hidden_markov_models, n_observation_classes, centroids, number_per_class):

    samplings = []

    pen_down_label = n_observation_classes - settings.PEN_DOWN_LABEL_DELTA
    pen_up_label = n_observation_classes - settings.PEN_UP_LABEL_DELTA
    stop_label = n_observation_classes - settings.STOP_LABEL_DELTA

    for hidden_markov_model in hidden_markov_models:

        model_samplings = []

        for i in range(0, number_per_class):

            encoded_stop_label = hidden_markov_model.label_encoder.transform(stop_label)
            observation_sequence, state_sequence = sample_hidden_markov_model(hidden_markov_model.hidden_markov_model, encoded_stop_label)
            observation_array = np.array(observation_sequence)

            sample_observations = hidden_markov_model.label_encoder.inverse_transform(observation_array)

            curves = []
            current_curve = []
            for observation in sample_observations:
                if observation < pen_down_label:
                    current_curve.append(centroids[observation])
                elif observation == pen_up_label:
                    if len(current_curve) > 0:
                        curves.append(current_curve)
                    current_curve = []
            if len(current_curve) > 0:
                curves.append(current_curve)

            dig = digit.Digit()
            for curve in curves:
                dig.add_curve(curve)

            model_samplings.append(dig)

        samplings.append(model_samplings)

    return samplings


def sample_hidden_markov_model(hidden_markov_model, encoded_stop_label):

    state_sequence = []
    observation_sequence = []

    current_state = get_start_state(hidden_markov_model)
    current_observation = get_observation(hidden_markov_model, current_state)

    state_sequence.append(current_state)
    observation_sequence.append(current_observation)

    found_stop = False

    while not found_stop and len(observation_sequence) < settings.MAX_SAMPLE_LENGTH:

        current_state = get_next_state(hidden_markov_model, current_state)
        current_observation = get_observation(hidden_markov_model, current_state)

        state_sequence.append(current_state)
        observation_sequence.append(current_observation)

        if current_observation == encoded_stop_label:
            found_stop = True

    return (observation_sequence, state_sequence)


def get_start_state(hidden_markov_model):

    startprob = hidden_markov_model.startprob_

    val = np.random.ranf()
    counter = 0.0

    index = 0
    while index < len(startprob) and counter < val:
        counter += startprob[index]
        if index < len(startprob) and counter < val:
            index += 1

    return index


def get_observation(hidden_markov_model, state):

    emitmat = hidden_markov_model.emissionprob_

    val = np.random.ranf()
    counter = 0.0

    index = 0
    while index < len(emitmat[state]) and counter < val:
        counter += emitmat[state][index]
        if index < len(emitmat[state]) and counter < val:
            index += 1

    return index


def get_next_state(hidden_markov_model, state):

    transmat = hidden_markov_model.transmat_

    val = np.random.ranf()
    counter = 0.0

    index = 0
    while index < len(transmat[state]) and counter < val:
        counter += transmat[state][index]
        if index < len(transmat[state]) and counter < val:
            index += 1

    return index
