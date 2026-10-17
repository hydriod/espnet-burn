use burn::prelude::*;
use burn_store::PytorchStore;

use crate::{
    error::Error,
    modules::hifi_gan::generator::{HiFiGANGenerator, HiFiGANGeneratorConfig},
};

#[derive(Module, Debug)]
pub enum VITSDecoder<B: Backend> {
    HiFiGANGenerator(HiFiGANGenerator<B>),
}

impl<B: Backend> VITSDecoder<B> {
    pub fn init_hifi_gan(
        in_channels: usize,
        out_channels: usize,
        channels: usize,
        global_channels: usize,
        kernel_size: usize,
        upsample_scales: Vec<usize>,
        upsample_kernel_sizes: Vec<usize>,
        resblock_kernel_sizes: Vec<usize>,
        resblock_dilations: Vec<Vec<usize>>,
        device: &B::Device,
    ) -> Result<Self, Error> {
        Ok(Self::HiFiGANGenerator(
            HiFiGANGeneratorConfig::new(
                in_channels,
                out_channels,
                channels,
                global_channels,
                kernel_size,
                upsample_scales,
                upsample_kernel_sizes,
                resblock_kernel_sizes,
                resblock_dilations,
            )
            .init(device)?,
        ))
    }
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            VITSDecoder::HiFiGANGenerator(m) => m.forward(x, None),
        }
    }

    pub fn key_remap(&mut self, store: PytorchStore, parents: &Vec<&str>) -> PytorchStore {
        match self {
            VITSDecoder::HiFiGANGenerator(m) => m.key_remap(store, parents),
        }
    }
}
