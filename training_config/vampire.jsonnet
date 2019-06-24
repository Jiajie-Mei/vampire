local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_DEVICE"));

local BASE_READER(LAZY, COVARIATE_TRAIN_FILE, COVARIATE_DEV_FILE) = {
  "lazy": LAZY == 1,
  "type": "vampire_reader",
  "covariates": {
    "year": std.extVar("YEAR_FILES"),
    "month": std.extVar("MONTH_FILES"),
    "day": std.extVar("DAY_FILES")
  }
};

{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": BASE_READER(std.parseInt(std.extVar("LAZY_DATASET_READER")), std.extVar("COVARIATE_TRAIN_FILE"), std.extVar("COVARIATE_DEV_FILE")),
   "validation_dataset_reader": BASE_READER(std.parseInt(std.extVar("LAZY_DATASET_READER")), std.extVar("COVARIATE_TRAIN_FILE"), std.extVar("COVARIATE_DEV_FILE")),
   "train_data_path": std.extVar("TRAIN_PATH"),
   "validation_data_path": std.extVar("DEV_PATH"),
   "vocabulary": {
      "type": "extended_vocabulary",
      "directory_path": std.extVar("VOCABULARY_DIRECTORY")
   },
   "model": {
      "type": "vampire",
      "bow_embedder": {
         "type": "bag_of_word_counts",
         "vocab_namespace": "vampire",
         "ignore_oov": true
      },
      "kl_weight_annealing": std.extVar("KL_ANNEALING"),
      "sigmoid_weight_1": std.extVar("SIGMOID_WEIGHT_1"),
      "sigmoid_weight_2": std.extVar("SIGMOID_WEIGHT_2"),
      "linear_scaling": std.extVar("LINEAR_SCALING"),
      "reference_counts": std.extVar("REFERENCE_COUNTS"),
      "reference_vocabulary": std.extVar("REFERENCE_VOCAB"),
      "update_background_freq": std.parseInt(std.extVar("UPDATE_BACKGROUND_FREQUENCY")) == 1,
      "track_npmi": std.parseInt(std.extVar("TRACK_NPMI")) == 1,
      "metadata_predictors": ["year_labels"],
      "vae": {
         "z_dropout": std.extVar("Z_DROPOUT"),
         "encoder": {
            "activations": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.extVar("ENCODER_ACTIVATION")),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.parseInt(std.extVar("VAE_HIDDEN_DIM"))),
            "input_dim": std.parseInt(std.extVar("VOCAB_SIZE")) + 1 + 52,
            "num_layers": std.parseInt(std.extVar("NUM_ENCODER_LAYERS"))
         },
         "mean_projection": {
            "activations": std.extVar("MEAN_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("VAE_HIDDEN_DIM"))),
            "input_dim": std.extVar("VAE_HIDDEN_DIM"),
            "num_layers": std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS"))
         },
        "log_variance_projection": {
            "activations": std.extVar("LOG_VAR_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("VAE_HIDDEN_DIM"))),
            "input_dim": std.parseInt(std.extVar("VAE_HIDDEN_DIM")),
            "num_layers": std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS"))
         },
         "decoder": {
            "activations": "linear",
            "hidden_dims": [std.parseInt(std.extVar("VOCAB_SIZE")) + 1],
            "input_dim": std.parseInt(std.extVar("VAE_HIDDEN_DIM")),
            "num_layers": 1
         },
         "type": "logistic_normal"
      }
   },
    "iterator": {
      "batch_size": std.parseInt(std.extVar("BATCH_SIZE")),
      "track_epoch": true,
      "type": "basic"
   },
   "trainer": {
      "cuda_device": CUDA_DEVICE,
      "num_serialized_models_to_keep": 1,
      "num_epochs": 50,
      "patience": 5,
      "optimizer": {
         "lr": std.extVar("LEARNING_RATE"),
         "type": "adam"
      },
      "validation_metric": std.extVar("VALIDATION_METRIC")
   }
}
