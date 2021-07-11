local embedding_dim = 768;
local bert_model = 'bert-base-uncased';
{
  dataset_reader: {
    type: 'text_classification_json',
    tokenizer: {
      type: 'pretrained_transformer',
      model_name: bert_model,
    },
    token_indexers: {
      bert: {
        type: 'pretrained_transformer',
        max_length: 512,
        model_name: bert_model,
      },
    },
  },
  train_data_path: 'data/20newsgroups/train.jsonl',
  validation_data_path: 'data/20newsgroups/validation.jsonl',
  test_data_path: 'data/20newsgroups/test.jsonl',
  model: {
    type: 'cvdd',
    anomaly_label: 'comp',
    primary_token_indexer: 'bert',
    namespace: 'tags',
    text_field_embedder: {
      token_embedders: {
        bert: {
          type: 'pretrained_transformer',
          max_length: 512,
          model_name: bert_model,
          train_parameters: false,
        },
      },
    },
    context_encoder: {
      type: 'pass_through',
      input_dim: embedding_dim,
    },
    attention_encoder: {
      type: 'feedforward',
      feedforward: {
        input_dim: embedding_dim,
        num_layers: 2,
        hidden_dims: [150, 5],
        activations: ['tanh', 'linear'],
      },
    },
    distance: 'cosine',
  },
  data_loader: {
    type: 'simple',
    batch_size: 64,
    shuffle: true,
  },
  trainer: {
    callbacks: ['mlflow'],
    optimizer: {
      type: 'adam',
      lr: 0.01,
    },
    learning_rate_scheduler: {
      type: 'step',
      step_size: 40,
      gamma: 0.1,
    },
    validation_metric: '+auc',
    num_epochs: 100,
    grad_norm: 10.0,
    patience: 10,
    cuda_device: -1,
  },
}
