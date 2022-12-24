use serde::{Serialize, Deserialize};


#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TransferFunction
{
    Nothing,
    Relu,
    LeakyRelu,
    Tanh,
    Sigmoid,
    Sigmoid2
}

impl TransferFunction
{
    #[inline(always)]
    pub fn t_f(&self, n: f64) -> f64
    {
        match self
        {
            TransferFunction::Nothing => n,
            TransferFunction::Relu => n.max(0.0),
            TransferFunction::LeakyRelu => n.max(0.01),
            TransferFunction::Tanh => n.tanh(),
            TransferFunction::Sigmoid => 0.5 + 0.5 * (n * 0.5).tanh(),
            TransferFunction::Sigmoid2 => 1.7159 * (0.66666666*n).tanh()
        }
    }

    //dtf? funy
    #[inline(always)]
    pub fn dt_f(&self, n: f64) -> f64
    {
        match self
        {
            TransferFunction::Nothing => 1.0,
            TransferFunction::Relu => if n>0.0 {1.0} else {0.0},
            TransferFunction::LeakyRelu => if n>0.0 {1.0} else {0.01},
            TransferFunction::Tanh => 1.0 - n.tanh().powi(2),
            TransferFunction::Sigmoid =>
            {
                0.25 - 0.25 * (n * 0.5).tanh().powi(2)
            },
            TransferFunction::Sigmoid2 =>
            {
                1.1427894 - 1.1427894 * (0.66666666*n).tanh().powi(2)
            }
        }
    }
}
