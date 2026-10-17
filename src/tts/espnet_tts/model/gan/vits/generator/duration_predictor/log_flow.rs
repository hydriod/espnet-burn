use burn::prelude::*;

/// Log flow module.
#[derive(Module, Debug, Clone)]
pub struct LogFlow;

impl LogFlow {
    pub fn forward<B: Backend>(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        inverse: bool,
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let eps = 1e-5;
        if !inverse {
            let y = x.clamp_min(eps).log() * x_mask;
            let logdet = -y.clone().sum_dims_squeeze(&[1, 2]);
            (y, logdet)
        } else {
            let x = x.exp() * x_mask;
            let device = x.device();
            let b = x.dims()[0];
            (x, Tensor::zeros([b], &device))
        }
    }
}
