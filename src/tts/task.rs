use std::{fs::File, marker::PhantomData, path::Path};

use burn::prelude::*;
use burn_store::{ModuleSnapshot, PytorchStore};

use crate::error::Error;
use crate::tts::config::Config;
use crate::tts::espnet_tts::{
    energy_extract::EnergyExtractChoice, feats_extract::FeatsExtractChoice, model::TTSChoice,
    normalize::NormalizeChoice, pitch_extract::PitchExtractChoice, ESPnetTTSModel,
};

pub struct TTSTask<B: Backend> {
    _marker: PhantomData<B>,
}

impl<B: Backend> TTSTask<B> {
    pub fn build_model_from_file(
        config_file: Option<&Path>,
        model_file: Option<&Path>,
        device: &B::Device,
    ) -> Result<(ESPnetTTSModel<B>, Config), Error> {
        let config_file = if config_file.is_none() {
            model_file
                .ok_or(Error::Argument(
                    "`config_file` and `model_file` is `None`.".to_string(),
                ))?
                .parent()
                .ok_or(Error::Argument(
                    "`model_file` is `\"/\"` or `\"\"`".to_string(),
                ))?
                .join("config.yaml")
        } else {
            config_file.unwrap().to_path_buf()
        };
        println!("config file: {}", config_file.display());

        let f = File::open(config_file)?;

        let args = yaml_serde::from_reader::<File, Config>(f)?;
        let mut model = Self::build_model(&args, device)?;

        if let Some(model_file) = model_file {
            let weight_store = PytorchStore::from_file(model_file);
            let mut weight_store = model.key_remap(weight_store, &vec![]).allow_partial(true);
            model
                .load_from(&mut weight_store)
                .map_err(|e| Error::Other(format!("failed to load model parameters: {e}")))?;
        }

        Ok((model, args))
    }

    pub fn build_model(args: &Config, device: &B::Device) -> Result<ESPnetTTSModel<B>, Error> {
        // 1. feats_extract
        let (feats_extract, odim) = if args.odim.is_none() {
            let extract = FeatsExtractChoice::<B>::new(
                &args.feats_extract,
                args.feats_extract_conf.as_ref().unwrap(),
                device,
            );
            let odim = extract.output_size();
            (extract, odim)
        } else {
            (
                FeatsExtractChoice::<B>::None(PhantomData),
                args.odim.unwrap() as usize,
            )
        };

        // 2. Normalization layer
        let normalize = if args.normalize.is_some() {
            panic!("TODO: Not implement ChoiceNormalize.");
            // NormalizeChoice::<B>::None(PhantomData)
        } else {
            NormalizeChoice::<B>::None(PhantomData)
        };

        // 3. TTS
        let tts = TTSChoice::new(
            &args.tts,
            args.token_list.len(),
            odim,
            args.tts_conf.clone(),
            device,
        )?;

        // 4. Extra components
        // COEIROINK で使用しないため優先度低
        let pitch_extract = if args.pitch_extract.is_some() {
            panic!("TODO: Build pitch_extract");
            // PitchExtractChoice::<B>::None(PhantomData)
        } else {
            PitchExtractChoice::<B>::None(PhantomData)
        };
        let energy_extract = if args.energy_extract.is_some() {
            panic!("TODO: Build energy_extract");
            // EnergyExtractChoice::<B>::None(PhantomData)
        } else {
            EnergyExtractChoice::<B>::None(PhantomData)
        };
        let pitch_normalize = if args.pitch_normalize.is_some() {
            panic!("TODO: Build pitch_normalize");
            // NormalizeChoice::<B>::None(PhantomData)
        } else {
            NormalizeChoice::<B>::None(PhantomData)
        };
        let energy_normalize = if args.energy_normalize.is_some() {
            panic!("TODO: Build energy_normalize");
            // NormalizeChoice::<B>::None(PhantomData)
        } else {
            NormalizeChoice::<B>::None(PhantomData)
        };

        Ok(ESPnetTTSModel::new(
            feats_extract,
            pitch_extract,
            energy_extract,
            normalize,
            pitch_normalize,
            energy_normalize,
            tts,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::flex::Flex;

    #[test]
    fn loads_model_from_checkpoint_and_config() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tmp/model");
        let config_file = model_dir.join("config.yaml");
        let model_file = model_dir.join("100epoch.pth");
        let device = Default::default();

        TTSTask::<Flex>::build_model_from_file(Some(&config_file), Some(&model_file), &device)
            .unwrap_or_else(|error| panic!("failed to load test model: {error}"));
    }
}
