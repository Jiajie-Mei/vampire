from typing import Dict
import numpy as np
import torch
from allennlp.models.model import Model
from modules.vae import VAE
from allennlp.data import Vocabulary
from allennlp.nn import InitializerApplicator
from overrides import overrides
from allennlp.training.metrics import CategoricalAccuracy, Average
from allennlp.modules import FeedForward


@Model.register("vae_classifier")
class VAE_CLF(Model):
    """
    Perform text classification with a VAE

    Params
    ______

    vocab: ``Vocabulary``
        vocabulary
    vae : ``VAE``
        variational autoencoder (RNN or BOW-based)
    classifier: ``FeedForward``
        feedforward network classifying input
    """
    def __init__(self, 
                 vocab: Vocabulary,
                 vae: VAE,
                 classifier: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super(VAE_CLF, self).__init__(vocab)
        self.metrics = {
            'kld': Average(),
            'reconstruction': Average(),
            'nll': Average(),
            'accuracy': CategoricalAccuracy(),
            'elbo': Average(),
        }
        self._num_labels = vocab.get_vocab_size("labels")
        self._vae = vae
        self._classifier = classifier
        self._classifier_loss = torch.nn.CrossEntropyLoss()
        self._output_logits = torch.nn.Linear(self._classifier.get_output_dim(), self._num_labels)
        initializer(self)

    @overrides
    def forward(self, tokens, label):  # pylint: disable=W0221
        """
        Given tokens and labels, generate document representation with
        a latent code and classify.
        """

        # run VAE to decode with a latent code
        vae_output = self._vae(tokens, label)

        if self._vae.__class__.__name__ == 'RNN_VAE':
            decoded_output = vae_output['decoded_output']
            document_vectors = torch.max(decoded_output, 1)[0]
        else:
            document_vectors = vae_output['decoded_output'].squeeze(0)

        # classify
        output = self._classifier(document_vectors)
        logits = self._output_logits(output)
        classifier_loss = self._classifier_loss(logits, label)


        # set metrics
        reconstruction_loss = vae_output['reconstruction']
        elbo = vae_output['elbo']
        kld = vae_output['kld']
        nll = vae_output['nll']
        self.metrics['accuracy'](logits, label)
        self.metrics["reconstruction"](reconstruction_loss.mean())
        self.metrics["elbo"](elbo.mean())
        self.metrics["kld"](kld.mean())
        self.metrics["nll"](nll.mean())
        # create clf_output
        clf_output = vae_output
        clf_output['loss'] = vae_output['elbo'] + classifier_loss

        return clf_output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: float(metric.get_metric(reset)) for metric_name, metric in self.metrics.items()}
