use burn::prelude::*;

#[derive(Module, Debug)]
pub enum PosteriorEncoder<B: Backend> {
    WavenetBase(PhantomData<B>),    // dummy
}
