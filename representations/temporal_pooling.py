
def aggregate_features(embedding, method, representation_level):

    if method == 'mean':
        if representation_level in ('FeatureExtractor', 'FeatureAggregator'):
            aggregated_features = embedding.mean(dim=2)
        if representation_level in ('TransformerLayer', 'LastHiddenState'):
            aggregated_features = embedding.mean(dim=1)

    if method == 'max':
        if representation_level in ('FeatureExtractor', 'FeatureAggregator'):
            aggregated_features, _ = embedding.max(dim=2)
        if representation_level in ('TransformerLayer', 'LastHiddenState'):
            aggregated_features, _ = embedding.max(dim=1)

    aggregated_features = aggregated_features.detach().numpy().flatten().tolist()
    
    return aggregated_features


