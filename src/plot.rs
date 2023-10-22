#![allow(unused)]

use std::ops::Range;

use array_math::*;
use plotters::{prelude::*, element::PointCollection, coord::ranged3d::{ProjectionMatrixBuilder, ProjectionMatrix}};

type T = f64;

const PLOT_RES: (u32, u32) = (1024, 760);
const PLOT_CAPTION_FONT: (&str, u32) = ("sans", 20);
const PLOT_MARGIN: u32 = 5;
const PLOT_LABEL_AREA_SIZE: u32 = 30;

//const NUM_LIMIT_RANGE_FRACTION: f64 = 0.00000000000001;
const NUM_LIMIT_RANGE: Range<f64> = -256.0..256.0;

pub fn color_fade_atan(hue: Range<T>) -> impl Fn(T, T) -> HSLColor
{
    move |u, v: T| {
        let mut t = {
            let mut theta = v.atan2(u);
            assert!(theta.is_finite(), "Theta is not finite");
            while theta < 0.0
            {
                theta += core::f64::consts::TAU;
            }
            (theta as f64/core::f64::consts::TAU) % 1.0
        };

        if hue.start % 1.0 != hue.end % 1.0
        {
            t *= 2.0;
            if t > 1.0
            {
                t = 2.0 - t;
            }
        }

        let c = (hue.start + (hue.end - hue.start)*t) % 1.0;

        HSLColor(c, 1.0, 0.5)
    }
}

fn isometric(mut pb: ProjectionMatrixBuilder) -> ProjectionMatrix
{
    pb.yaw = core::f64::consts::FRAC_PI_4;
    pb.pitch = core::f64::consts::FRAC_PI_4;
    pb.scale = 0.7;
    pb.into_matrix()
}

pub fn plot_curve<const N: usize>(
    plot_title: &str, plot_path: &str,
    x: [T; N],
    y: [T; N]
) -> Result<(), Box<dyn std::error::Error>>
{
    let x_min = x.reduce(T::min).unwrap();
    let x_max = x.reduce(T::max).unwrap();
    
    let y_min = y.reduce(T::min).unwrap();
    let y_max = y.reduce(T::max).unwrap();
    
    let area = BitMapBackend::new(plot_path, PLOT_RES).into_drawing_area();
    
    area.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&area)
        .caption(plot_title, PLOT_CAPTION_FONT.into_font())
        .margin(PLOT_MARGIN)
        .x_label_area_size(PLOT_LABEL_AREA_SIZE)
        .y_label_area_size(PLOT_LABEL_AREA_SIZE)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
    
    chart.configure_mesh()
        .set_all_tick_mark_size(0.1)
        .draw()?;
    
    chart.draw_series(LineSeries::new(
            x.zip2(y),
            &BLUE
        ))?;
        
    // To avoid the IO failure being ignored silently, we manually call the present function
    area.present().expect("Unable to write result to file");

    Ok(())
}

pub fn plot_curve_2d<const NX: usize, const NY: usize, const P: usize>(
    plot_title: &str, plot_path: &str,
    x: [T; NX],
    y: [T; NY],
    f: [(&str, impl Fn(T, T) -> T); P]
) -> Result<(), Box<dyn std::error::Error>>
{
    plot_parametric_curve_2d(plot_title, plot_path,
        x, y,
        f.map(|f| (f.0, move |x, y| [x, y, f.1(x, y)]))
    )
}

pub fn plot_curve_2d_styled<C, const NX: usize, const NY: usize, const P: usize>(
    plot_title: &str, plot_path: &str,
    x: [T; NX],
    y: [T; NY],
    f: [(&str, impl Fn(T, T) -> (T, C)); P]
) -> Result<(), Box<dyn std::error::Error>>
where
    C: Color + Copy
{
    plot_parametric_curve_2d_styled(plot_title, plot_path,
        x, y,
        f.map(|f| (f.0, move |x, y| {
            let (z, c) = f.1(x, y);
            ([x, y, z], c)
        }))
    )
}

pub fn plot_parametric_curve_2d<const NU: usize, const NV: usize, const P: usize>(
    plot_title: &str, plot_path: &str,
    u: [T; NU],
    v: [T; NV],
    f: [(&str, impl Fn(T, T) -> [T; 3]); P]
) -> Result<(), Box<dyn std::error::Error>>
{
    let hues: [Range<T>; P] = ArrayOps::fill(|i| {
        (i as f64/P as f64)..((i as f64 + 1.0)/P as f64)
    });

    plot_parametric_curve_2d_styled(
        plot_title,
        plot_path,
        u,
        v,
        f.zip2(hues.each_ref2())
            .map(|((name, f), hue)| (name, move |u, v| (f(u, v), color_fade_atan(hue.clone())(u, v))))
    )
}

pub fn plot_parametric_curve_2d_styled<C, const NU: usize, const NV: usize, const P: usize>(
    plot_title: &str, plot_path: &str,
    u: [T; NU],
    v: [T; NV],
    f: [(&str, impl Fn(T, T) -> ([T; 3], C)); P]
) -> Result<(), Box<dyn std::error::Error>>
where
    C: Color + Copy
{
    use plotters::prelude::*;

    let area = SVGBackend::new(plot_path, PLOT_RES).into_drawing_area();
    
    let f_ref = f.each_ref2();
    let f_values: Vec<([T; 3], [T; 3])> = u.into_iter()
        .flat_map(|u| v.into_iter()
            .map(move |v| f_ref.map(|(_, f_ref)| {
                    let fuv: [T; 3] = f_ref(u, v).0;
                    (fuv, fuv)
                }).reduce(|a, b| (a.0.comap(b.0, T::min), a.0.comap(b.0, T::max)))
                .unwrap_or_else(|| ([0.0; 3], [0.0; 3]))
            )
        ).collect();

    let ([x_min, y_min, z_min], [x_max, y_max, z_max]) = f_values.into_iter()
        .reduce(|a, b| (a.0.zip2(b.0).map(|(a, b)| a.min(b)), a.1.zip2(b.1).map(|(a, b)| a.max(b))))
        .unwrap();

    area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&area)
        .caption(plot_title, PLOT_CAPTION_FONT)
        .set_all_label_area_size(PLOT_LABEL_AREA_SIZE)
        .build_cartesian_3d(x_min..x_max, z_min..z_max, y_min..y_max)?;

    chart.with_projection(isometric);
    
    chart.configure_axes()
        .light_grid_style(BLACK.mix(0.15))
        .max_light_lines(3)
        .draw()?;
    
    for (f_name, f) in f
    {
        let mut count_tot = 0;
        let (mut r_tot, mut g_tot, mut b_tot, mut a_tot): (u128, u128, u128, f64) = Default::default();
        chart.draw_series(
                SurfaceSeries::xoz(
                    u.into_iter(),
                    v.into_iter(),
                    f,
                )
                .map(|polygon| {
                    let (mut r, mut g, mut b, mut a): (u128, u128, u128, f64) = Default::default();
                    let points: Vec<(T, T, T)> = polygon.point_iter()
                        .into_iter()
                        .map(|&(u, ([x, y, z], color), v)| {
                            let color = color.to_rgba();
                            r += color.0 as u128;
                            g += color.1 as u128;
                            b += color.2 as u128;
                            a += color.3;
                            (x, z, y)
                        })
                        .collect();
    
                    let count = points.len() as u128;
    
                    let color = RGBAColor(
                        (r/count) as u8,
                        (g/count) as u8,
                        (b/count) as u8,
                        a/(count as f64)
                    );
    
                    count_tot += 1;
                    r_tot += color.0 as u128;
                    g_tot += color.1 as u128;
                    b_tot += color.2 as u128;
                    a_tot += color.3;
    
                    Polygon::new(points, color.mix(0.5).filled())
                })
            )?
            .label(f_name)
            .legend({
                let color = RGBAColor(
                    (r_tot/count_tot) as u8,
                    (g_tot/count_tot) as u8,
                    (b_tot/count_tot) as u8,
                    a_tot/(count_tot as f64)
                );
                move |(x, y)| Rectangle::new([(x + 5, y - 5), (x + 15, y + 5)], color.mix(0.5).filled())
            });
    }
    
    chart.configure_series_labels()
        .border_style(&BLACK)
        .draw()?;
    
    // To avoid the IO failure being ignored silently, we manually call the present function
    area.present().expect("Unable to write result to file");

    Ok(())
}

pub fn plot_curve_2d_rad<const NTHETA: usize, const NR: usize>(
    plot_title: &str, plot_path: &str,
    r: [T; NR],
    theta: [T; NTHETA],
    f: impl Fn(T, T) -> T
) -> Result<(), Box<dyn std::error::Error>>
where
    [(); 2*NR]:
{
    use plotters::prelude::*;

    let area = SVGBackend::new(plot_path, PLOT_RES).into_drawing_area();

    let r_max = r.into_iter().map(|r| r.abs()).reduce(T::max).unwrap();
    
    let theta_min = theta.into_iter().map(|theta| theta.abs()).reduce(T::min).unwrap();
    let theta_max = theta.into_iter().map(|theta| theta.abs()).reduce(T::max).unwrap();

    let f_ref = &f;
    let f_values: Vec<T> = r.into_iter().flat_map(|r| theta.into_iter().map(move |theta| f_ref(r, theta))).collect();
    let (z_min, z_max) = f_values.into_iter().map(|f| (f, f)).reduce(|a, b| (a.0.min(b.0), a.1.max(b.1))).unwrap();

    area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&area)
        .caption(plot_title, PLOT_CAPTION_FONT)
        .set_all_label_area_size(PLOT_LABEL_AREA_SIZE)
        .build_cartesian_3d(-r_max..r_max, z_min..z_max, -r_max..r_max)?;

    chart.with_projection(isometric);
    
    chart.configure_axes()
        .light_grid_style(BLACK.mix(0.15))
        .max_light_lines(3)
        .draw()?;

    chart.draw_series(
            SurfaceSeries::xoz(
                r.into_iter(),
                theta.into_iter(),
                |r, theta| f(r, theta),
            )
            //.style_func(&|&c| HSLColor(c as f64, 1.0, 0.5).mix(0.2).filled())
            .map(|polygon| {
                let mut sum_theta = 0.0;
                let points: Vec<(T, T, T)> = polygon.point_iter()
                    .into_iter()
                    .map(|&(r, z, theta)| {sum_theta += theta; (r*theta.cos(), z, r*theta.sin())})
                    .collect();
                let avg_theta = sum_theta / points.len() as T;
                let c = (((avg_theta - theta_min)/(theta_max - theta_min)) as f64 + 1.0) % 1.0;
                Polygon::new(points, HSLColor(c, 1.0, 0.5).mix(0.2).filled())
            })
        )?
        .label("Radial surface")
        .legend(|(x, y)| Rectangle::new([(x + 5, y - 5), (x + 15, y + 5)], BLUE.mix(0.5).filled()));
    
    chart.configure_series_labels()
        .border_style(&BLACK)
        .draw()?;
    
    // To avoid the IO failure being ignored silently, we manually call the present function
    area.present().expect("Unable to write result to file");

    Ok(())
}

pub fn plot_parametric_curve_2d_rad<const NU: usize, const NV: usize>(
    plot_title: &str, plot_path: &str,
    u: [T; NU],
    v: [T; NV],
    f: impl Fn(T, T) -> [T; 3]
) -> Result<(), Box<dyn std::error::Error>>
{
    use plotters::prelude::*;

    let area = SVGBackend::new(plot_path, PLOT_RES).into_drawing_area();

    let f_ref = &f;
    let f_values: Vec<[T; 3]> = u.into_iter().flat_map(|u| v.into_iter().map(move |v| f_ref(u, v))).collect();

    let ([_r_min, theta_min, z_min], [r_max, theta_max, z_max]) = f_values.into_iter()
        .map(|f| (f, f))
        .reduce(|a, b| (a.0.zip2(b.0).map(|(a, b)| a.min(b)), a.1.zip2(b.1).map(|(a, b)| a.max(b))))
        .unwrap();

    area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&area)
        .caption(plot_title, PLOT_CAPTION_FONT)
        .set_all_label_area_size(PLOT_LABEL_AREA_SIZE)
        .build_cartesian_3d(-r_max..r_max, z_min..z_max, -r_max..r_max)?;

    chart.with_projection(isometric);
    
    chart.configure_axes()
        .light_grid_style(BLACK.mix(0.15))
        .max_light_lines(3)
        .draw()?;

    chart.draw_series(
            SurfaceSeries::xoz(
                u.into_iter(),
                v.into_iter(),
                |u, v| f(u, v),
            )
            //.style_func(&|&c| HSLColor(c as f64, 1.0, 0.5).mix(0.2).filled())
            .map(|polygon| {
                let mut sum_theta = 0.0;
                let points: Vec<(T, T, T)> = polygon.point_iter()
                    .into_iter()
                    .map(|&(_, [r, theta, z], _)| {sum_theta += theta; (r*theta.cos(), z, r*theta.sin())})
                    .collect();
                let avg_theta = sum_theta / points.len() as T;
                let c = (((avg_theta - theta_min)/(theta_max - theta_min)) as f64 + 1.0) % 1.0;
                Polygon::new(points, HSLColor(c, 1.0, 0.5).mix(0.2).filled())
            })
        )?
        .label("Radial surface")
        .legend(|(x, y)| Rectangle::new([(x + 5, y - 5), (x + 15, y + 5)], BLUE.mix(0.5).filled()));
    
    chart.configure_series_labels()
        .border_style(&BLACK)
        .draw()?;
    
    // To avoid the IO failure being ignored silently, we manually call the present function
    area.present().expect("Unable to write result to file");

    Ok(())
}