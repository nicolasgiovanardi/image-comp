import unittest
import torch
from src.train import CompressionModel # This will now work because 'src' is the top-level package

class TestCompressionModel(unittest.TestCase):
    def setUp(self):
        self.model = CompressionModel(128, 64, 6, 32, 16)
        self.batch_size = 4
        self.input_shape = (1, 28, 28)
        self.dummy_input = torch.randn(self.batch_size, *self.input_shape)

    def test_output_shape(self):
        """Verify that reconstructed image matches input dimensions."""
        output = self.model(self.dummy_input)
        # Check if it's a tuple and get the first element (reconstruction)
        recon_image = output[0] if isinstance(output, tuple) else output
        self.assertEqual(recon_image.shape, self.dummy_input.shape)

    def test_gradient_flow(self):
        """Verify that backpropagation reaches the encoder weights."""
        output = self.model(self.dummy_input)
        # If tuple, sum the reconstruction or the total loss provided by the model
        val_to_backprop = output[0].sum() if isinstance(output, tuple) else output.sum()
        val_to_backprop.backward()
        
        # Check gradients
        has_grads = any(p.grad is not None and p.grad.sum() != 0 for p in self.model.parameters())
        self.assertTrue(has_grads, "No gradients detected. Check if modules are connected.")

if __name__ == "__main__":
    unittest.main()
