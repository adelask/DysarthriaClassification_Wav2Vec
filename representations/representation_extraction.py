import torch
import representations.temporal_pooling as temporal_pooling

def extract_features(model, model_label, representation_level, transformer_layer, pooling, audio_input):

    embedding = None

    with torch.no_grad():

        if model_label == 'w2v1':
            embedding = model.feature_extractor(audio_input)

            if representation_level == 'FeatureAggregator':
                embedding = model.feature_aggregator(embedding)

        if model_label == 'w2v2':
            
            if representation_level == 'FeatureExtractor':
                embedding = model.feature_extractor(audio_input)

            if representation_level == 'LastHiddenState':
                embedding = model(audio_input).last_hidden_state

            if representation_level == 'TransformerLayer':
                if transformer_layer is False:
                    raise ValueError("transformer_layer must be specified when using 'TransformerLayer'")
                embedding = model(audio_input, output_hidden_states=True).hidden_states[transformer_layer]

    if embedding is None:
        raise ValueError(f"Could not compute embedding for model {model_label} and representation_level {representation_level}")

    # pooling
    out = temporal_pooling.aggregate_features(embedding, pooling, representation_level)

    return out
