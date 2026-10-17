use burn::prelude::*;
use burn::tensor::activation::{relu, selu, silu, tanh};

pub fn forward_activation<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    activation: &str,
) -> Tensor<B, D> {
    match activation {
        "tanh" => tanh(x),
        "silu" => selu(x),
        "relu" => relu(x),
        "swish" => silu(x),
        _ => panic!("Not implemented activation."),
    }
}
