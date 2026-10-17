use std::marker::PhantomData;

use burn::prelude::*;

#[derive(Module, Debug)]
pub enum NormalizeChoice<B: Backend> {
    None(PhantomData<B>),
}

impl<B: Backend> NormalizeChoice<B> {
    /// Mirrors `AbsNormalize.forward(feats, feats_lengths)`.
    pub fn forward<const D: usize>(
        &self,
        _feats: Tensor<B, D>,
        _feats_lengths: Tensor<B, 1>,
    ) -> (Tensor<B, D>, Tensor<B, 1>) {
        match self {
            Self::None(_) => panic!("NormalizeChoice::None does not implement `forward`."),
        }
    }

    /// Mirrors `InversibleInterface.inverse(feats)`.
    pub fn inverse<const D: usize>(&self, _feats: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Self::None(_) => panic!("NormalizeChoice::None does not implement `inverse`."),
        }
    }
}
