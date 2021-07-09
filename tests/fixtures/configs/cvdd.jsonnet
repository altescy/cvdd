{
  dataset_reader: {
    type: 'text_classification_json',
  },
  train_data_path: 'tests/fixtures/data/imdb_corpus.jsonl',
  validation_data_path: 'tests/fixtures/data/imdb_corpus.jsonl',
  model: {
    type: 'cvdd',
    anomaly_label: 'pos',
    text_field_embedder: {
      token_embedders: {
        tokens: {
          type: 'embedding',
          embedding_dim: 10,
          trainable: true,
        },
      },
    },
    context_encoder: {
      type: 'pass_through',
      input_dim: 10,
    },
    attention_encoder: {
      type: 'feedforward',
      feedforward: {
        input_dim: 10,
        num_layers: 2,
        hidden_dims: [10, 3],
        activations: ['tanh', 'linear'],
        biases: false,
      },
    },
    distance: 'cosine',
  },
  data_loader: {
    type: 'simple',
    batch_size: 2,
    shuffle: false,
  },
  trainer: {
    optimizer: {
      type: 'adam',
      lr: 0.01,
    },
    validation_metric: '+auc',
    num_epochs: 3,
    grad_norm: 10.0,
    patience: 5,
    cuda_device: -1,
  },
}
