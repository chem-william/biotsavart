extern crate itertools;
extern crate ndarray;
extern crate simdeez;

use itertools::Itertools;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use ndarray::prelude::*;
use ndarray::Zip;

use simdeez::sse2::*;
use simdeez::*;
use std::*;

fn convert(phi: Vec<Vec<Vec<f64>>>) -> Array3<f64> {
    let flattened: Vec<f64> = phi.concat().concat();
    let init = Array3::from_shape_vec((phi.len(), phi[0].len(), phi[0][0].len()), flattened);
    init.unwrap()
}

// Calculates the magnetization in accordance to the paper
// "Local Current Density Calculations for Molecular Films from Ab Initio"
// DOI: 10.1021/acs.jctc.5b00471
// J. Chem. Theory Comput. 2015, 11, 5161−5176

// TODO: in the paper it uses the negative of the current density. I don't. Is that bad?
fn calculate_magnetization(
    center: Array1<f64>,
    jx: &Array3<f64>,
    jy: &Array3<f64>,
    jz: &Array3<f64>,
    x_cor: &[f64],
    y_cor: &[f64],
    z_cor: &[f64],
) -> Vec<f64> {
    let mut temp_x = Array3::<f64>::zeros(jx.dim());
    let mut temp_y = Array3::<f64>::zeros(jy.dim());
    let mut temp_z = Array3::<f64>::zeros(jz.dim());

    Zip::indexed(&mut temp_x)
        .and(&mut temp_y)
        .and(&mut temp_z)
        .par_apply(|idx, result_x, result_y, result_z| {
            let b_r = array![
                x_cor[idx.0] as f64,
                y_cor[idx.1] as f64,
                z_cor[idx.2] as f64
            ] - &center;
            let jx_val = &jx[[idx.0, idx.1, idx.2]];
            let jy_val = &jy[[idx.0, idx.1, idx.2]];
            let jz_val = &jz[[idx.0, idx.1, idx.2]];

            *result_x = -b_r[1] * jz_val + jy_val * b_r[2];
            *result_y = b_r[0] * jz_val - jx_val * b_r[2];
            *result_z = -b_r[0] * jy_val + jx_val * b_r[1];
        });

    vec![temp_x.sum() * 0.5, temp_y.sum() * 0.5, temp_z.sum() * 0.5]
}

/// Calculates the magnetic field, B, generated by a current density, J
///
/// Parameters
/// ----------
/// center : ndarray
///     Array of x-, y-, z- coordinates where the magnetization is calculated
/// jx : ndarray
///     Values of Jx on a 3D grid. Has to be a matrix of size MxNxK.
/// jy : ndarray
///     Values of Jy on a 3D grid. Has to be a matrix of size MxNxK.
/// jz : ndarray
///     Values of Jz on a 3D grid. Has to be a matrix of size MxNxK.
/// x_cor : array_like
///     X coordinates for the first dimension of the J values grid.
/// y_cor : array_like
///     Y coordinates for the second dimension of the J values grid.
/// z_cor : array_like
///     Z coordinates for the third dimension of the J values grid.
///
/// Returns
/// -------
/// B : tuple of array_like
///     tuple of Bx, By and Bz. Each list has to be reshaped to match the original size of J.
#[pyfunction]
fn biot(
    center: Vec<f64>,
    jx: Vec<Vec<Vec<f64>>>,
    jy: Vec<Vec<Vec<f64>>>,
    jz: Vec<Vec<Vec<f64>>>,
    x_cor: Vec<f64>,
    y_cor: Vec<f64>,
    z_cor: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let center = Array1::from(center);
    let jx = convert(jx);
    let jy = convert(jy);
    let jz = convert(jz);

    let mut b_x = Array3::<f64>::zeros(jx.dim());
    let mut b_y = Array3::<f64>::zeros(jy.dim());
    let mut b_z = Array3::<f64>::zeros(jz.dim());

    println!("Calculating m..");
    let m_vec = calculate_magnetization(
        center,
        &jx,
        &jy,
        &jz,
        x_cor.as_slice(),
        y_cor.as_slice(),
        z_cor.as_slice(),
    );

    Zip::indexed(&mut b_x)
        .and(&mut b_y)
        .and(&mut b_z)
        .par_apply(|idx, result_x, result_y, result_z| {
            let b_r = [
                x_cor[idx.0] as f64,
                y_cor[idx.1] as f64,
                z_cor[idx.2] as f64,
            ];

            for (xi, x) in x_cor.iter().enumerate() {
                for (yi, y) in y_cor.iter().enumerate() {
                    let mut chunk_idx: usize = 0;
                    for z in &z_cor.iter().chunks(64) {
                        let a: Vec<f64> = z.cloned().collect();
                        let a_len = a.len();
                        let b = a.as_slice();

                        let jx = jx
                            .slice(s![xi, yi, chunk_idx..chunk_idx + a_len])
                            .to_slice()
                            .unwrap();
                        let jy = jy
                            .slice(s![xi, yi, chunk_idx..chunk_idx + a_len])
                            .to_slice()
                            .unwrap();
                        let jz = jz
                            .slice(s![xi, yi, chunk_idx..chunk_idx + a_len])
                            .to_slice()
                            .unwrap();

                        let res_temp = sum_compiletime(&b_r, *x, *y, b, jx, jy, jz);
                        *result_x += res_temp.0.iter().filter(|val| !val.is_nan()).sum::<f64>();
                        *result_y += res_temp.1.iter().filter(|val| !val.is_nan()).sum::<f64>();
                        *result_z += res_temp.2.iter().filter(|val| !val.is_nan()).sum::<f64>();

                        chunk_idx += a_len;
                    }
                }
            }
        });

    Ok((
        b_x.into_raw_vec(),
        b_y.into_raw_vec(),
        b_z.into_raw_vec(),
        m_vec,
    ))
}

simd_compiletime_generate!(
    fn sum(
        b_r: &[f64; 3],
        x: f64,
        y: f64,
        z: &[f64],
        jx: &[f64],
        jy: &[f64],
        jz: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let x_arr = [x; 64];
        let y_arr = [y; 64];
        let rx_arr = [b_r[0]; 64];
        let ry_arr = [b_r[1]; 64];
        let rz_arr = [b_r[2]; 64];

        let mut res_x: Vec<f64> = Vec::with_capacity(z.len());
        let mut res_y: Vec<f64> = Vec::with_capacity(z.len());
        let mut res_z: Vec<f64> = Vec::with_capacity(z.len());
        res_x.set_len(z.len());
        res_y.set_len(z.len());
        res_z.set_len(z.len());

        for i in (0..z.len()).step_by(S::VF64_WIDTH) {
            let xv1 = S::loadu_pd(&x_arr[i]);
            let yv1 = S::loadu_pd(&y_arr[i]);
            let zv1 = S::loadu_pd(&z[i]);

            let brx_v1 = S::loadu_pd(&rx_arr[i]);
            let bry_v1 = S::loadu_pd(&ry_arr[i]);
            let brz_v1 = S::loadu_pd(&rz_arr[i]);

            let jx = S::loadu_pd(&jx[i]);
            let jy = S::loadu_pd(&jy[i]);
            let jz = S::loadu_pd(&jz[i]);

            let rx = brx_v1 - xv1;
            let ry = bry_v1 - yv1;
            let rz = brz_v1 - zv1;

            let distance = S::sqrt_pd((rx * rx) + (ry * ry) + (rz * rz));
            let r3 = distance * distance * distance;

            let result_x = (jy * rz - ry * jz) / r3;
            let result_y = (rx * jz - jx * rz) / r3;
            let result_z = (jx * ry - rx * jy) / r3;

            S::storeu_pd(&mut res_x[i], result_x);
            S::storeu_pd(&mut res_y[i], result_y);
            S::storeu_pd(&mut res_z[i], result_z);
        }
        (res_x, res_y, res_z)
    }
);

#[pymodule]
fn libbiot_savart(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(biot))?;

    Ok(())
}
