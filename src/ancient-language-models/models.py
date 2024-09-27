from transformers.modeling_outputs import TokenClassifierOutput
from transformers import RobertaPreTrainedModel, RobertaConfig, RobertaModel
from torch import nn
import torch


class DependencyRobertaForTokenClassification(RobertaPreTrainedModel):
    config_class = RobertaConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.u_a = nn.Linear(768, 768)
        self.w_a = nn.Linear(768, 768)
        self.v_a_inv = nn.Linear(768, 1, bias=False)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        return_dict=True,
        **kwargs,
    ):
        output = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )[0]
        batch_size, seq_len, hidden_size = output.size()

        source = output.unsqueeze(2).expand(-1, -1, seq_len, -1)
        target = output.unsqueeze(1).expand(-1, seq_len, -1, -1)
        function_g = self.v_a_inv(
            torch.tanh(self.u_a(source) + self.w_a(target))
        ).squeeze(-1)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
            function_g = function_g.masked_fill(mask == 0, -1e4)
        p_head = nn.functional.log_softmax(function_g, dim=2)

        loss = None
        if labels is not None:
            loss_fct = nn.NLLLoss(ignore_index=-100)
            loss = loss_fct(p_head.view(-1, seq_len), labels.view(-1))

        if not return_dict:
            output = (p_head,)
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=p_head,
            hidden_states=None,
            attentions=None,
        )

    def attention(self, source, target, mask=None):
        function_g = self.v_a_inv(torch.tanh(self.u_a(source) + self.w_a(target)))
        if mask is not None:
            function_g.masked_fill_(mask, -1e4)
        return nn.functional.log_softmax(function_g, dim=1)
