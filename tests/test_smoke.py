"""CPU smoke tests requiring no patient data or pretrained checkpoints."""

import unittest

import torch

from src.model import Transformer
from src.DR_Learner.train_nuisance_models import SingleHeadTreatmentEffectModel
from src.DR_Learner.train_pseudo_outcome_cate_model import PseudoOutcomeCateModel


class ModelSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 3
        self.seq_length = 6
        self.vocab_size = 32
        self.d_model = 16
        self.encoder = Transformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=2,
            num_layers=1,
            ff_mult=2,
            dropout=0.0,
        ).eval()

    def test_transformer_forward(self):
        tokens = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_length))
        ages = 40.0 + torch.arange(self.seq_length).float().expand(self.batch_size, -1)
        lengths = torch.tensor([6, 4, 1])
        padding_mask = torch.arange(self.seq_length).unsqueeze(0) >= lengths.unsqueeze(1)
        tokens[padding_mask] = 0
        ages = ages.clone()
        ages[padding_mask] = 0.0

        with torch.no_grad():
            outputs = self.encoder(tokens, ages, key_padding_mask=padding_mask)

        expected_shapes = {
            "logits": (self.batch_size, self.seq_length, self.vocab_size),
            "age_preds": (self.batch_size, self.seq_length),
            "hidden": (self.batch_size, self.seq_length, self.d_model),
            "pooled_logits": (self.batch_size, self.vocab_size),
            "pooled_age": (self.batch_size,),
        }
        self.assertEqual(set(outputs), set(expected_shapes))
        for name, shape in expected_shapes.items():
            with self.subTest(output=name):
                self.assertEqual(tuple(outputs[name].shape), shape)
                self.assertTrue(torch.isfinite(outputs[name]).all().item())

    def test_nuisance_head(self):
        model = SingleHeadTreatmentEffectModel(
            self.encoder, head_hidden_dim=8, head_dropout=0.0
        ).eval()
        embeddings = torch.randn(self.batch_size, self.d_model)
        with torch.no_grad():
            logits = model.head(embeddings)
        # Propensity, comparator outcome, and treated outcome logits.
        self.assertEqual(tuple(logits.shape), (self.batch_size, 3))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_cate_head(self):
        model = PseudoOutcomeCateModel(
            self.encoder, head_hidden_dim=8, head_dropout=0.0
        ).eval()
        embeddings = torch.randn(self.batch_size, self.d_model)
        with torch.no_grad():
            predictions = model.head(embeddings)
        self.assertEqual(tuple(predictions.shape), (self.batch_size, 1))
        self.assertTrue(torch.isfinite(predictions).all().item())
        self.assertEqual(tuple(predictions.squeeze(-1).shape), (self.batch_size,))


if __name__ == "__main__":
    unittest.main()
