from transformers import GPT2LMHeadModel

from datatuner.lm.cross_entropy import CrossEntropyLoss


def custom_gpt2_with_smoothing(smoothing=0.0):
    class GPT2LMHeadModelCustom(GPT2LMHeadModel):
        def forward(
            self,
            input_ids,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None,
        ):

            transformer_outputs = self.transformer(
                input_ids, past=past, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask
            )

            hidden_states = transformer_outputs[0]

            lm_logits = self.lm_head(hidden_states)

            outputs = (lm_logits,) + transformer_outputs[1:]
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(ignore_index=-1, smooth_eps=smoothing, reduction="mean")

                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                outputs = (loss,) + outputs

            return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

    return GPT2LMHeadModelCustom
