use burn::prelude::*;
use burn_store::PytorchStore;

use crate::{
    error::Error,
    modules::residual_coupling::{ResidualAffineCouplingBlock, ResidualAffineCouplingBlockConfig},
};

#[derive(Module, Debug)]
pub enum VITSFlow<B: Backend> {
    ResidualCoupling(ResidualAffineCouplingBlock<B>),
}

impl<B: Backend> VITSFlow<B> {
    pub fn init_residual_coupling(
        in_channels: usize,
        hidden_channels: usize,
        flows: usize,
        kernel_size: usize,
        base_dilation: usize,
        layers: usize,
        global_channels: isize,
        dropout_rate: f64,
        _use_weight_norm: bool,
        use_only_mean: bool,
        device: &B::Device,
    ) -> Result<Self, Error> {
        let m = ResidualAffineCouplingBlockConfig::new(in_channels, hidden_channels, flows)
            .with_kernel_size(kernel_size)
            .with_base_dilation(base_dilation)
            .with_layers(layers)
            .with_global_channels(global_channels)
            .with_dropout_rate(dropout_rate)
            .with_use_only_mean(use_only_mean)
            .init(device)?;

        Ok(Self::ResidualCoupling(m))
    }
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        g: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        match self {
            VITSFlow::ResidualCoupling(m) => m.forward(x, x_mask, g, false),
        }
    }

    pub fn inverse(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        g: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        match self {
            VITSFlow::ResidualCoupling(m) => m.forward(x, x_mask, g, true),
        }
    }

    pub fn key_remap(&mut self, store: PytorchStore, parents: &Vec<&str>) -> PytorchStore {
        match self {
            VITSFlow::ResidualCoupling(m) => m.key_remap(store, parents),
        }
    }
}
