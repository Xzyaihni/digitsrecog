use std::{
    slice,
    ffi::{CStr, c_char}
};

use neural_net::*;

mod neural_net;


#[repr(C)]
pub struct Guesses
{
    guesses: [f64;10]
}

#[no_mangle]
pub extern "C" fn recognize(network_path: *const c_char, image: *const u8) -> Guesses
{
    if network_path.is_null() || image.is_null()
    {
        Guesses{guesses: [0.0; 10]}
    } else
    {
        let network_path = unsafe{ CStr::from_ptr(network_path) };
        let network_path = network_path.to_str().unwrap();

        let image = unsafe{ slice::from_raw_parts(image, 28*28) };

        let mut network = NeuralNet::load(network_path)
            .map_err(|err| format!("{err} (filepath: {network_path})")).unwrap();

        let guesses = network.feedforward(&image.iter().map(|v| *v as f64 / 255.0)
            .collect::<Vec<f64>>());

        Guesses{guesses: guesses.clone().try_into().unwrap()}
    }
}