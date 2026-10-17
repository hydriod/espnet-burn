pub mod energy_extract;
pub mod feats_extract;
pub mod model;
pub mod normalize;
pub mod pitch_extract;

use burn::prelude::*;
use burn_store::PytorchStore;

use self::energy_extract::EnergyExtractChoice;
use self::feats_extract::FeatsExtractChoice;
use self::model::{TTSChoice, TTSSynthesisOptions, TTSSynthesisOutput};
use self::normalize::NormalizeChoice;
use self::pitch_extract::PitchExtractChoice;
use crate::error::Error;

#[derive(Module, Debug)]
pub struct ESPnetTTSModel<B: Backend> {
    pub feats_extract: FeatsExtractChoice<B>,
    pub pitch_extract: PitchExtractChoice<B>,
    pub energy_extract: EnergyExtractChoice<B>,
    pub normalize: NormalizeChoice<B>,
    pub pitch_normalize: NormalizeChoice<B>,
    pub energy_normalize: NormalizeChoice<B>,
    pub tts: TTSChoice<B>,
}

impl<B: Backend> ESPnetTTSModel<B> {
    pub fn new(
        feats_extract: FeatsExtractChoice<B>,
        pitch_extract: PitchExtractChoice<B>,
        energy_extract: EnergyExtractChoice<B>,
        normalize: NormalizeChoice<B>,
        pitch_normalize: NormalizeChoice<B>,
        energy_normalize: NormalizeChoice<B>,
        tts: TTSChoice<B>,
    ) -> Self {
        ESPnetTTSModel {
            feats_extract,
            pitch_extract,
            energy_extract,
            normalize,
            pitch_normalize,
            energy_normalize,
            tts,
        }
    }

    pub fn key_remap(&mut self, store: PytorchStore, parent: &Vec<&str>) -> PytorchStore {
        let mut parent = parent.clone();
        parent.push("tts");
        let store = self.tts.key_remap(store, &parent);
        parent.pop();
        store
    }

    pub fn inference(
        &self,
        text: Tensor<B, 1, Int>,
        speech: Option<Tensor<B, 1>>,
        spembs: Option<Tensor<B, 1>>,
        sids: Option<Tensor<B, 1>>,
        lids: Option<Tensor<B, 1>>,
        durations: Option<Tensor<B, 1>>,
        pitch: Option<Tensor<B, 1>>,
        energy: Option<Tensor<B, 1>>,
        options: TTSSynthesisOptions,
        use_teacher_forcing: bool,
    ) -> Result<TTSSynthesisOutput<B>, Error> {
        if use_teacher_forcing {
            let speech = speech.ok_or_else(|| {
                Error::Argument("missing required argument: 'speech'".to_string())
            })?;

            match &self.feats_extract {
                FeatsExtractChoice::None(_) => {
                    // `feats = speech`: normalize the raw speech directly.
                    if !matches!(self.normalize, NormalizeChoice::None(_)) {
                        let device = speech.device();
                        let wav_len = speech.dims()[0];
                        let lengths = Tensor::<B, 1>::from_data([wav_len as f32], &device);
                        let _ = self
                            .normalize
                            .forward(speech.unsqueeze_dim::<2>(0), lengths);
                    }
                }
                feats_extract => {
                    let device = speech.device();
                    let wav_len = speech.dims()[0];
                    // `speech[None]`: (T_wav,) -> (Batch=1, Nsamples, Channels=1)
                    let input: Tensor<B, 3> = speech.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(2);
                    let input_lengths = Tensor::<B, 1>::from_data([wav_len as f32], &device);
                    let (feats, feats_lengths) = feats_extract
                        .forward(input, input_lengths)
                        .map_err(|e| Error::Other(e.to_string()))?;
                    // `[0]`: drop the batch dim added above.
                    let [_, frames, channels, freq] = feats.dims();
                    let feats: Tensor<B, 3> = feats
                        .slice([0..1, 0..frames, 0..channels, 0..freq])
                        .squeeze_dim(0);
                    if !matches!(self.normalize, NormalizeChoice::None(_)) {
                        let (feats, _) = self
                            .normalize
                            .forward(feats.unsqueeze_dim::<4>(0), feats_lengths);
                        let [_, frames, channels, freq] = feats.dims();
                        let _: Tensor<B, 3> = feats
                            .slice([0..1, 0..frames, 0..channels, 0..freq])
                            .squeeze_dim(0);
                    }
                }
            }
        }

        if use_teacher_forcing {
            let _ = (durations, pitch, energy);
            return Err(Error::Argument(
                "use_teacher_forcing=true is not supported yet: the teacher-forcing path \
                 (pitch/energy extraction, posterior encoder) is not ported."
                    .to_string(),
            ));
        }

        let _ = (spembs, sids, lids);

        let output = self.tts.inference(text, options, use_teacher_forcing)?;

        match self.normalize {
            NormalizeChoice::None(_) => {},
        }

        Ok(output)
    }
}
