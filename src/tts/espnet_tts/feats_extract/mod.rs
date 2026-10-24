pub mod linear_spectrogram;

use std::marker::PhantomData;

use burn::{prelude::*, tensor::DataError};

use self::linear_spectrogram::LinearSpectrogram;
use crate::tts::config::FeatsExtractConfig;

#[derive(Module, Debug)]

pub enum FeatsExtractChoice<B: Backend> {
    LinearSpectrogram(LinearSpectrogram<B>),
    None(PhantomData<B>),
}

impl<B: Backend> FeatsExtractChoice<B> {
    pub fn new(feat_extract: &str, config: &FeatsExtractConfig, device: &B::Device) -> Self {
        match feat_extract {
            "linear_spectrogram" => Self::LinearSpectrogram(LinearSpectrogram::new(
                Some(config.n_fft),
                config.win_length,
                Some(config.hop_length),
                config.window.as_deref(),
                config.center,
                config.normalized,
                config.onesided,
                device,
            )),
            _ => Self::None(PhantomData),
        }
    }

    pub fn output_size(&self) -> usize {
        match self {
            Self::LinearSpectrogram(feats_extract) => feats_extract.output_size(),
            Self::None(PhantomData) => 0,
        }
    }

    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        input_lengths: Tensor<B, 1>,
    ) -> Result<(Tensor<B, 4>, Tensor<B, 1>), DataError> {
        match self {
            Self::LinearSpectrogram(feats_extract) => feats_extract.forward(input, input_lengths),
            Self::None(PhantomData) => {
                panic!("FeatsExtractChoice::None does not implement `forward`.")
            }
        }
    }
}
