use burn::module::Param;
use burn::prelude::*;

/// Elementwise affine flow module.
#[derive(Module, Debug)]
pub struct ElementwiseAffineFlow<B: Backend> {
    m: Param<Tensor<B, 2>>,
    logs: Param<Tensor<B, 2>>,
}

impl<B: Backend> ElementwiseAffineFlow<B> {
    pub fn new(channels: usize, device: &B::Device) -> Self {
        let m = Param::from_tensor(Tensor::zeros([channels, 1], device));
        let logs = Param::from_tensor(Tensor::zeros([channels, 1], device));
        Self { m, logs }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        inverse: bool,
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let m = self.m.val().unsqueeze_dim(0);
        let logs = self.logs.val().unsqueeze_dim(0);

        if !inverse {
            let y = (m + logs.clone().exp() * x) * x_mask.clone();
            let logdet = (logs * x_mask).sum_dims_squeeze(&[1, 2]);
            (y, logdet)
        } else {
            let x = (x - m) * (-logs).exp() * x_mask;
            let device = x.device();
            let b = x.dims()[0];
            (x, Tensor::zeros([b], &device))
        }
    }
}
