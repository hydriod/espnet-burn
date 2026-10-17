use std::marker::PhantomData;

use burn::prelude::*;

#[derive(Module, Debug)]
pub enum PitchExtractChoice<B: Backend> {
    Dio(PhantomData<B>),
    None(PhantomData<B>),
}
