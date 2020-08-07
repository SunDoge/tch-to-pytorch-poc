use dlpack::*;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use tch::{kind, Device, Kind, Tensor};
use lazy_static::lazy_static;
use std::ffi::{CString, CStr, c_void};

lazy_static! {
    static ref NAME: CString = CString::new("dltensor").unwrap();
}

fn get_strides_from_shape(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];

    let mut c = 1;
    strides[shape.len() - 1] = c;
    for i in (1..shape.len()).rev() {
        c *= shape[i];
        strides[i - 1] = c;
    }

    strides
}

fn tensor_to_dltensor(x: &Box<Tensor>) -> DLTensor {
    let data = x.data_ptr();
    let mut device_id = 0;
    let device_type = match x.device() {
        Device::Cpu => DLDeviceType::DLCPU,
        Device::Cuda(i) => {
            device_id = i as i32;
            DLDeviceType::DLGPU
        }
    };
    let ctx = DLContext {
        device_type,
        device_id,
    };
    let ndim = x.dim() as i32;
    // let dtype = DLDataType {
    //     code:
    // }
    let dtype: DLDataType = match x.kind() {
        Kind::Int64 => (DLDataTypeCode::DLInt, 64, 1),
        Kind::Float => (DLDataTypeCode::DLFloat, 32, 1),
        _ => unimplemented!(),
    }
    .into();

    let mut shape = x.size();
    let mut strides = get_strides_from_shape(&shape);
    let byte_offset = 0;

    let dlt = DLTensor {
        data,
        ctx,
        ndim,
        dtype,
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset,
    };

    std::mem::forget(shape);
    std::mem::forget(strides);

    dlt
}

unsafe extern "C" fn deleter(x: *mut DLManagedTensor) {
    println!("DLManagedTensor deleter");

    let ctx = (*x).manager_ctx as *mut Tensor;
    std::mem::drop(ctx);
}

unsafe extern "C" fn destructor(o: *mut pyo3::ffi::PyObject) {
    println!("PyCapsule destructor");


    // let ptr = pyo3::ffi::PyCapsule_GetPointer(o, name.as_ptr()) as *mut dlpack::DLManagedTensor;

    let ptr = pyo3::ffi::PyCapsule_GetName(o);
    let current_name = CStr::from_ptr(ptr);
    println!("Current Name: {:?}", current_name);

    if current_name != NAME.as_c_str() {
        return;
    }

    let ptr = pyo3::ffi::PyCapsule_GetPointer(o, NAME.as_ptr()) as *mut dlpack::DLManagedTensor;
    (*ptr).deleter.unwrap()(ptr);

    println!("Delete by Python");

    // dbg!(*ptr);
    // (*ptr).deleter.unwrap()(ptr);
}

#[pyfunction]
fn eye(n: i64) -> PyResult<*mut pyo3::ffi::PyObject> {
    let x = Tensor::eye(n, kind::FLOAT_CPU);
    let bx = Box::new(x);
    let dlt = tensor_to_dltensor(&bx);
    // dbg!(dlt);
    let dlmt = DLManagedTensor {
        dl_tensor: dlt,
        manager_ctx: Box::into_raw(bx) as *mut c_void,
        deleter: Some(deleter)
    };

    let bdlmt = Box::new(dlmt);

    let ptr = unsafe {
        pyo3::ffi::PyCapsule_New(
            Box::into_raw(bdlmt) as *mut c_void,
            NAME.as_ptr(),
            Some(destructor as pyo3::ffi::PyCapsule_Destructor),
        )
    };

    Ok(ptr)
}

/// A Python module implemented in Rust.
#[pymodule]
fn tch(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(eye))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
