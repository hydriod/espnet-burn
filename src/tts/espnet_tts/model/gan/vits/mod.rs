mod generator;

use burn::prelude::*;
use burn_store::PytorchStore;

use crate::{error::Error, tts::config::TTSConig};

use self::generator::VITSGenerator;
pub use self::generator::VITSSynthesisOptions;

pub struct VITSConfig {
    pub idim: usize,
    pub odim: usize,
    pub tts_conf: TTSConig,
}
impl VITSConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<VITS<B>, Error> {
        let generator = match self.tts_conf.generator_type.as_str() {
            "vits_generator" => self.tts_conf.generator_params.init_vits_generator(device)?,
            other => panic!("Not implement {other}."),
        };

        Ok(VITS { generator })
    }
}

#[derive(Module, Debug)]
pub struct VITS<B: Backend> {
    generator: VITSGenerator<B>,
}
impl<B: Backend> VITS<B> {
    pub fn inference(
        &self,
        text: Tensor<B, 1, Int>,
        options: VITSSynthesisOptions,
        use_teacher_forcing: bool,
    ) -> Result<VITSynthesisOutput<B>, Error> {
        if use_teacher_forcing {
            return Err(Error::Argument(
                "use_teacher_forcing=true requires the posterior encoder, which is not ported yet."
                    .to_string(),
            ));
        }

        let device = text.device();
        let text_length = text.dims()[0];
        let text = text.unsqueeze_dim(0);
        let text_length = Tensor::<B, 1>::from_data([text_length as f32], &device);
        let gen_output =
            self.generator
                .inference(text, text_length, options, use_teacher_forcing)?;

        Ok(VITSynthesisOutput {
            generated: gen_output.generated.flatten(0, -1),
            attention_weight: gen_output.attention_weight.squeeze_dim(0),
            duration: gen_output.duration.squeeze_dim(0),
        })
    }

    pub fn key_remap(&mut self, store: PytorchStore, parents: &Vec<&str>) -> PytorchStore {
        let mut parents = parents.clone();
        parents.push("generator");
        self.generator.key_remap(store, &parents)
    }
}

#[derive(Clone, Debug)]
pub struct VITSynthesisOutput<B: Backend> {
    pub generated: Tensor<B, 1>,
    pub attention_weight: Tensor<B, 2>,
    pub duration: Tensor<B, 1>,
}
