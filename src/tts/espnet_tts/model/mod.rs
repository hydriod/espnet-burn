pub mod gan;
pub mod utils;

use std::fmt::{Display, Formatter};
use std::marker::PhantomData;

use burn::prelude::*;
use burn_store::PytorchStore;

use crate::{error::Error, tts::config::TTSConig};

use self::gan::vits::{VITSConfig, VITSSynthesisOptions, VITSynthesisOutput, VITS};

#[derive(Module, Debug)]
pub enum TTSChoice<B: Backend> {
    VITS(VITS<B>),
    None(PhantomData<B>),
}

impl<B: Backend> TTSChoice<B> {
    pub fn new(
        tts: impl AsRef<str>,
        vocab_size: usize,
        odim: usize,
        tts_conf: Option<TTSConig>,
        device: &B::Device,
    ) -> Result<Self, Error> {
        let tts_class = tts.as_ref();
        let tts = match tts_class {
            "vits" => {
                let tts_conf = tts_conf.ok_or(Error::Argument(
                    "tts_conf is required for vits.".to_string(),
                ))?;
                let vits = VITSConfig {
                    idim: vocab_size,
                    odim,
                    tts_conf,
                }
                .init(device)?;
                TTSChoice::VITS(vits)
            }
            _ => TTSChoice::None(PhantomData),
        };
        Ok(tts)
    }

    pub fn inference(
        &self,
        text: Tensor<B, 1, Int>,
        options: TTSSynthesisOptions,
        use_teacher_forcing: bool,
    ) -> Result<TTSSynthesisOutput<B>, Error> {
        match (self, options) {
            (TTSChoice::VITS(vits), TTSSynthesisOptions::VITS(options)) => {
                let output = vits.inference(text, options, use_teacher_forcing)?;
                Ok(TTSSynthesisOutput::VITS(output))
            }
            (TTSChoice::None(_), _) => Err(Error::Argument(
                "TTSChoice is None, cannot perform inference.".to_string(),
            )),
            /*
            (tts, options) => Err(Error::Argument(format!(
                "Mismatch between TTSChoice ({tts}) and TTSSynthesisOptions ({options})."
            ))),
            */
        }
    }

    pub fn key_remap(&mut self, store: PytorchStore, parents: &Vec<&str>) -> PytorchStore {
        match self {
            TTSChoice::VITS(vits) => vits.key_remap(store, parents),
            TTSChoice::None(_) => store,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TTSSynthesisOptions {
    VITS(VITSSynthesisOptions),
}

impl TTSSynthesisOptions {
    pub fn new_vits(alpha: f64, noise_scale: f64, noise_scale_dur: f64) -> Self {
        Self::VITS(VITSSynthesisOptions {
            alpha,
            noise_scale,
            noise_scale_dur,
        })
    }
}

impl Display for TTSSynthesisOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VITS(options) => write!(f, "VITS({options:?})"),
        }
    }
}

#[derive(Debug)]
pub enum TTSSynthesisOutput<B: Backend> {
    VITS(VITSynthesisOutput<B>),
}
