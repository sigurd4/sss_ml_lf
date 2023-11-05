use std::ops::AddAssign;

use num::Float;

use super::*;

#[derive(Clone, Copy, Debug)]
pub struct GeneralizedSmoothHingeLoss
{
    pub alpha: f64
}

impl GeneralizedSmoothHingeLoss
{
    pub fn new(alpha: f64) -> Self
    {
        Self {
            alpha
        }
    }
}

impl<F, const Y: usize> LossFunction<F, Y, Y, Y> for GeneralizedSmoothHingeLoss
where
    F: Float + AddAssign + Default,
    [(); Y - Y]:
{
    fn lf_loss(&self, y_true: [F; Y], y_est: [F; Y]) -> [F; Y]
    {
        let zero = F::zero();
        let one = F::one();
        let alpha = f!(self.alpha; F);

        y_true.comap(y_est, |y_true, y_est| {
            let z = y_true*y_est;
            if z >= one
            {
                zero
            }
            else if z > zero
            {
                (alpha + z.powf(alpha + one))/(alpha + one) - z
            }
            else
            {
                alpha/(alpha + one) - z
            }
        })
    }
    fn lf_loss_grad(&self, y_true: [F; Y], y_est: [F; Y]) -> [[F; Y]; Y]
    {
        let zero = F::zero();
        let one = F::one();
        let alpha = f!(self.alpha; F);

        y_true.comap(y_est, |y_true, y_est| {
            let z = y_true*y_est;
            if z >= one
            {
                zero
            }
            else if z > zero
            {
                z.powf(alpha) - one
            }
            else
            {
                -one
            }
        })
            .diagonal()
    }
}

#[cfg(test)]
mod test
{
    use crate::tests as t;
    use super::GeneralizedSmoothHingeLoss as LF;

    #[test]
    fn test()
    {
        let alpha = 0.5;

        t::test(LF::new(alpha), [1.0, -1.0, 1.0, 1.0, -1.0], [0.6, -1.2, 0.8, 1.5, -1.3]);
        t::graph_2d(LF::new(alpha), [0.8, 0.2], -1.0..1.0);
    }
}