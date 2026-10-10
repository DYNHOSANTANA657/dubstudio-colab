"""Regression checks for the MADLAD loader; not a GPU/quality approval."""
import unittest
from types import SimpleNamespace

import dubstudio_translate as translator


class TranslationLoadingTests(unittest.TestCase):
    def test_lm_head_is_resolved_after_input_embeddings(self):
        class Model:
            _tied_weights_keys = {"lm_head.weight": "shared.weight", "encoder.embed_tokens.weight": "shared.weight", "decoder.embed_tokens.weight": "shared.weight"}
        self.assertTrue(translator.configure_madlad_weight_tying(Model))
        self.assertEqual(list(Model._tied_weights_keys), ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"])
        self.assertFalse(translator.configure_madlad_weight_tying(Model))

    def test_unrecognized_mapping_is_not_modified(self):
        for mapping in (["lm_head.weight"], {"lm_head.weight": "other.weight"}, None):
            class Model:
                _tied_weights_keys = mapping
            self.assertFalse(translator.configure_madlad_weight_tying(Model))
            self.assertIs(Model._tied_weights_keys, mapping)

    def test_corrupted_encoder_is_rejected_before_generation(self):
        tensors = SimpleNamespace(equal=lambda a, b: a == b)
        def model(encoder, decoder, head):
            return SimpleNamespace(encoder=SimpleNamespace(embed_tokens=SimpleNamespace(weight=encoder)), decoder=SimpleNamespace(embed_tokens=SimpleNamespace(weight=decoder)), lm_head=SimpleNamespace(weight=head))
        translator.validate_madlad_embeddings(model(250, 250, 2), tensors)
        for values in ((2, 250, 2), (250, 250, 250), (99, 250, 2)):
            with self.assertRaisesRegex(RuntimeError, "carregados incorretamente"):
                translator.validate_madlad_embeddings(model(*values), tensors)

    def test_complete_long_generation_is_not_rejected_by_length_alone(self):
        self.assertTrue(translator.generation_reached_eos([0] + [7] * 511 + [2], 2))
        self.assertFalse(translator.generation_reached_eos([0] + [7] * 512, 2))
        self.assertTrue(translator.generation_reached_eos([0, 7, 2, 1], [2, 3]))
        self.assertFalse(translator.generation_reached_eos([2, 7], 2))


if __name__ == "__main__":
    unittest.main()
