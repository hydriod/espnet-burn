use burn::prelude::*;

/// Flip flow module.
#[derive(Module, Debug, Clone)]
pub struct FlipFlow;

impl FlipFlow {
    pub fn forward<B: Backend>(
        &self,
        x: Tensor<B, 3>,
        inverse: bool,
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let c = x.dims()[1];
        let flipped_indices = Tensor::arange_step(0..c as i64, 1, &x.device()).flip([0]);
        let x = x.select(1, flipped_indices);
        let b = x.dims()[0];
        let device = x.device();

        if !inverse {
            (x, Tensor::zeros([b], &device))
        } else {
            (x, Tensor::zeros([b], &device))
        }
    }
}
