#![feature(associated_type_bounds)]
#![feature(decl_macro)]
#![feature(const_trait_impl)]

#![feature(generic_const_exprs)]

moddef::moddef!(
    flat(pub) mod {
        loss_function
    },
    mod {
        plot for cfg(test)
    }
);

use array_math::*;

#[cfg(test)]
mod tests
{
    use std::{fmt::Debug, ops::{Div, AddAssign}};

    use linspace::LinspaceArray;
    use num::NumCast;
    use num_identities_const::ZeroConst;
    use plotters::style::HSLColor;

    use super::*;

    pub fn test<T, F, const Y_TRUE: usize, const Y_EST: usize, const L: usize>(lf: T, y_true: [F; Y_TRUE], y_est: [F; Y_EST])
    where
        F: Copy + Debug + NumCast + Div<F, Output: Debug> + AddAssign<F> + ZeroConst,
        T: LossFunction<F, Y_TRUE, Y_EST, L>
    {
        let l = lf.lf_loss(y_true, y_est);
        
        let e = l.sum()/f!(L);
        
        println!("Error = {:?}", e);
    
        println!("Loss = {:?}", l);
        
        let l_grad = lf.lf_loss_grad(y_true, y_est);

        println!("Gradient of loss = {:?}", l_grad);
    }
    
    const N: usize = 64;
    const HUE_SATURATION: f64 = 1.618;
    const SHADOW_SATURATION: f64 = 0.5;
    const SHADOW_DIRECTION: [f64; 3] = [1.618, 1.0, 100.0].normalize_to(SHADOW_SATURATION);
    
    pub fn graph_2d<T, R, const Y_TRUE: usize, const L: usize>(lf: T, y_true: [f64; Y_TRUE], y_est_range: R)
    where
        T: LossFunction<f64, Y_TRUE, 2, L> + Debug,
        R: LinspaceArray<f64, N>
    {
        let y_est = y_est_range.linspace_array();
        
        let lf_name = format!("{:?}", lf);

        let mut first = true;
        let file_name: String = lf_name.chars()
            .flat_map(|c| {
                let c_low = c.to_ascii_lowercase();
                if c.is_uppercase() && !first
                {
                    vec!['_', c_low]
                }
                else
                {
                    first = false;
                    vec![c_low]
                }
            }).collect();

        let lf_ref = &lf;
        
        let lf_n = <[_; L]>::fill(|i| {
            //let hue = (i as f64/L as f64)..((i as f64 + 1.0)/L as f64);
            let hue_positive = 1.0/3.0..0.0;
            let hue_negative = 1.0/3.0..2.0/3.0;

            (format!("e{}", if L > 1 {format!("_{}", i)} else {"".to_string()}), move |y0, y1| {
                let l = lf_ref.lf_loss(y_true, [y0, y1])[i];
                let cx = 1.0 - (-l.abs()*HUE_SATURATION).exp2();
                let hue = if l >= 0.0 {&hue_positive} else {&hue_negative}.clone();
                let mut c = HSLColor((hue.start + (hue.end - hue.start)*cx) % 1.0, 1.0, 0.5);

                let e_grad = lf_ref.lf_loss_grad(y_true, [y0, y1])[i];
                let [u, v] = [[e_grad[0], 0.0, 1.0], [0.0, e_grad[1], 1.0]];
                let mut n = u.mul_cross([&v]);
                n[2] += f64::EPSILON;
                
                let shade: f64 = n.normalize()
                    .mul_dot(SHADOW_DIRECTION);

                c.2 = 0.5 + 0.5*(1.0 - (-shade.abs()).exp()).copysign(shade);

                (l, c)
            })
        });

        crate::plot::plot_curve_2d_styled(
            &format!("Plot of loss function '{}' for y = {:?}", lf_name, y_true),
            &format!("plot/lf_{}_e_2d.svg", file_name),
            y_est,
            y_est,
            lf_n.each_ref2()
                .map(|(name, f)| (name.as_str(), f))
        ).expect("Plot error");
    }
}

macro f
{
    ($n:expr; $f:ident) => {
        <$f>::from($n).unwrap()
    },
    ($n:expr) => {
        num::NumCast::from($n).unwrap()
    },
}