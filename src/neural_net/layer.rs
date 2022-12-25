use serde::{Serialize, Deserialize};
use rand::Rng;

pub use transfer::*;


mod transfer;


pub enum InnerOuter<'a>
{
    Outputs(&'a [f64]),
    Inners(&'a [f64], &'a [Vec<f64>])
}

type Sign = i8;
fn new_sign(num: f64) -> Sign
{
    if num==0.0
    {
        0
    } else if num>0.0
    {
        1
    } else
    {
        -1
    }
}

#[derive(Debug, Clone)]
pub struct DefaultLayerSettings
{
    pub size: usize,
    pub transfer_function: TransferFunction
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultLayer
{
    #[serde(skip)]
    neurons: Vec<f64>,

    learning_rates: Vec<Vec<f64>>,
    previous_signs: Vec<Vec<Sign>>,
    #[serde(skip)]
    gradient_batch: Vec<Vec<f64>>,
    weights: Vec<Vec<f64>>,



    transfer_function: TransferFunction
}

impl DefaultLayer
{
    pub fn new(size: usize, previous_size: usize, transfer_function: TransferFunction) -> Self
    {
        let neurons = (0..size).map(|_| 0.0).collect::<Vec<f64>>();

        let mut rng = rand::thread_rng();
        let weights = (0..size).map(|_|
        {
            //+1 for bias
            (0..previous_size+1).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect::<Vec<f64>>()
        }).collect::<Vec<Vec<f64>>>();

        let gradient_batch = weights.iter().map(|wc| vec![0.0; wc.len()])
            .collect::<Vec<Vec<f64>>>();
        let learning_rates = weights.iter().map(|wc| vec![0.1; wc.len()])
            .collect::<Vec<Vec<f64>>>();
        let previous_signs = weights.iter().map(|wc|
        {
            wc.iter().map(|w| new_sign(*w)).collect::<Vec<_>>()
        }).collect::<Vec<Vec<_>>>();

        DefaultLayer{
            neurons,
            learning_rates, previous_signs, gradient_batch,
            weights,
            transfer_function
        }
    }

    pub fn size(&self) -> usize
    {
        self.neurons.len()
    }

    pub fn neurons(&self) -> &[f64]
    {
        &self.neurons
    }

    pub fn weights(&self) -> &[Vec<f64>]
    {
        &self.weights
    }

    pub fn transfer_function(&self) -> TransferFunction
    {
        self.transfer_function
    }

    pub fn reset_temporary(&mut self)
    {
        self.neurons = (0..self.weights.len()).map(|_| 0.0).collect::<Vec<f64>>();

        self.gradient_batch = self.weights.iter().map(|wc| vec![0.0; wc.len()])
            .collect::<Vec<Vec<f64>>>();
    }

    pub fn feedforward(&mut self, previous_neurons: &[f64], transfer_function: TransferFunction)
    {
        self.neurons.iter_mut().zip(self.weights.iter()).for_each(|(neuron, neuron_weights)|
        {
            let bias = unsafe{ neuron_weights.get_unchecked(neuron_weights.len()-1) };

            *neuron = previous_neurons.iter()
                .zip(neuron_weights.iter()).map(|(previous_neuron, weight)|
                {
                    transfer_function.t_f(*previous_neuron) * *weight
                }).sum::<f64>() + bias;
        });
    }

    pub fn apply_gradients(&mut self)
    {
        for neuron in 0..self.weights.len()
        {
            let previous_length = unsafe{ self.weights.get_unchecked(neuron).len() };
            for previous in 0..previous_length
            {
                let (gradient, previous_sign, learning_rate, weight);
                unsafe
                {
                gradient = self.gradient_batch
                    .get_unchecked_mut(neuron)
                    .get_unchecked_mut(previous);

                previous_sign = self.previous_signs
                    .get_unchecked_mut(neuron)
                    .get_unchecked_mut(previous);

                learning_rate = self.learning_rates
                    .get_unchecked_mut(neuron)
                    .get_unchecked_mut(previous);
                
                weight = self.weights
                    .get_unchecked_mut(neuron)
                    .get_unchecked_mut(previous);
                }

                let current_sign = new_sign(*gradient);

                let combination = current_sign * *previous_sign;
                if combination>0
                {
                    *learning_rate = (*learning_rate * 1.2).min(0.01);

                    *weight -= *learning_rate * current_sign as f64;
                    *previous_sign = current_sign;
                } else if combination<0
                {
                    *learning_rate = (*learning_rate * 0.5).max(0.000001);

                    *previous_sign = 0;
                } else
                {
                    *weight -= *learning_rate * current_sign as f64;
                    *previous_sign = current_sign;
                }
    
                *gradient = 0.0;
            }
        }
    }

    pub fn combine(&mut self, other: &DefaultLayer)
    {
        for i_neuron in 0..self.gradient_batch.len()
        {
            unsafe
            {
            for i_previous in 0..self.gradient_batch.get_unchecked(i_neuron).len()
            {
                *self.gradient_batch.get_unchecked_mut(i_neuron).get_unchecked_mut(i_previous) +=
                    *other.gradient_batch.get_unchecked(i_neuron).get_unchecked(i_previous);
            }
            }
        }
    }

    pub fn backpropagate(
        &mut self,
        inputs: &[f64],
        errors: InnerOuter
    )
    {
        for i_neuron in 0..self.neurons.len()
        {
            let neuron = unsafe{ self.neurons.get_unchecked_mut(i_neuron) };

            let error = match errors
            {
                InnerOuter::Outputs(correct) =>
                {
                    unsafe
                    {
                    self.transfer_function.t_f(*neuron) - *correct.get_unchecked(i_neuron)
                    }
                },
                InnerOuter::Inners(neurons, weights) =>
                {
                    neurons.iter().zip(weights.iter()).map(|(next_neuron, next_weight)|
                    {
                        next_neuron * unsafe{ *next_weight.get_unchecked(i_neuron) }
                    }).sum::<f64>()
                }
            };

            let deriv = self.transfer_function.dt_f(*neuron) * error;

            let current_batch = unsafe{ self.gradient_batch.get_unchecked_mut(i_neuron) };

            inputs.iter().zip(current_batch.iter_mut()).for_each(|(input, gradient)|
            {
                *gradient += deriv * *input;
            });

            let last_gradient = current_batch.len()-1;
            //add bias gradient
            unsafe{ *current_batch.get_unchecked_mut(last_gradient) += deriv };

            //set current neuron to its derivative
            *neuron = deriv;
        }
    }
}


#[cfg(test)]
pub mod tests
{
    use super::*;

    pub fn get_weight(layer: &mut DefaultLayer, neuron: usize, previous: usize) -> &mut f64
    {
        layer.weights[neuron].get_mut(previous).unwrap()
    }

    pub fn get_gradient(layer: &mut DefaultLayer, neuron: usize, previous: usize) -> &mut f64
    {
        layer.gradient_batch[neuron].get_mut(previous).unwrap()
    }
}
