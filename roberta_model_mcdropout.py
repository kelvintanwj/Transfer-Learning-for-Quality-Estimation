from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import (
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    # RobertaClassificationHead, # making edits directly on RobertaClassificationHead
    RobertaConfig,
    RobertaModel,
)

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    
    def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train(inplace=True)
            
    def get_monte_carlo_predictions(data_loader, forward_passes,
                                    model, n_classes, n_samples):
        """ Function to get the monte-carlo samples and uncertainty estimates
        through multiple forward passes
        Parameters
        ----------
        data_loader : object
            data loader object from the data loader module
        forward_passes : int
            number of monte-carlo samples/forward passes
        model : object
            keras model
        n_classes : int
            number of classes in the dataset
        n_samples : int
            number of samples in the test set
        """
        dropout_predictions = np.empty((0, n_samples, n_classes))
        softmax = nn.Softmax(dim=1)
        for i in range(forward_passes):
            predictions = np.empty((0, n_classes))
            model.eval()
            enable_dropout(model)
            for i, (image, label) in enumerate(data_loader):

                image = image.to(torch.device('cuda'))
                with torch.no_grad():
                    output = model(image)
                    output = softmax(output) # shape (n_samples, n_classes)
                predictions = np.vstack((predictions, output.cpu().numpy()))

            dropout_predictions = np.vstack((dropout_predictions,
                                             predictions[np.newaxis, :, :]))
            # dropout predictions - shape (forward_passes, n_samples, n_classes)

        # Calculating mean across multiple MCD forward passes 
        mean = np.mean(dropout_predictions, axis=0) # shape (n_samples, n_classes)

        # Calculating variance across multiple MCD forward passes 
        variance = np.var(dropout_predictions, axis=0) # shape (n_samples, n_classes)

        epsilon = sys.float_info.min
        # Calculating entropy across multiple MCD forward passes 
        entropy = -np.sum(mean*np.log(mean + epsilon), axis=-1) # shape (n_samples,)

        # Calculating mutual information across multiple MCD forward passes 
        mutual_info = entropy - np.mean(np.sum(-dropout_predictions*np.log(dropout_predictions + epsilon),
                                                axis=-1), axis=0) # shape (n_samples,)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """  # noqa: ignore flake8"
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "roberta"

    def __init__(self, config, weight=None):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.weight = weight

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if self.weight is not None:
                    weight = self.weight.to(labels.device)
                else:
                    weight = None
                loss_fct = CrossEntropyLoss(weight=weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
