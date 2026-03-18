//! `bcs_rs` — high-performance Rust extension for BCS data decoding.

mod transport;
mod z85;

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Duration;

use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::conversion::IntoPyObject;
use pyo3::types::{PyBool, PyDict, PyFloat, PyList, PyString};
use serde_json::{Map, Value};

impl From<z85::Z85Error> for PyErr {
    fn from(err: z85::Z85Error) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

impl From<transport::BcsError> for PyErr {
    fn from(err: transport::BcsError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

fn python_to_serde_map(dict: &Bound<'_, PyDict>) -> PyResult<BTreeMap<String, Value>> {
    let mut out = BTreeMap::new();
    for (k, v) in dict.iter() {
        let key = k.extract::<String>().map_err(|_| {
            PyTypeError::new_err("dict keys must be strings")
        })?;
        let value = python_to_serde_value(v)?;
        out.insert(key, value);
    }
    Ok(out)
}

fn python_to_serde_value(obj: Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.extract()?));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Number(serde_json::Number::from(i)));
    }
    if let Ok(u) = obj.extract::<u64>() {
        return Ok(Value::Number(serde_json::Number::from(u)));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Value::Number(
            serde_json::Number::from_f64(f).unwrap_or(serde_json::Number::from(0)),
        ));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Value::String(s.extract()?));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let arr: Vec<Value> = list
            .iter()
            .map(|item| python_to_serde_value(item))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(Value::Array(arr));
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        let map = python_to_serde_map(&d)?;
        return Ok(Value::Object(Map::from_iter(map)));
    }
    if obj.hasattr("__iter__")? && !obj.is_instance_of::<PyString>() {
        let list: Py<PyList> = obj.extract()?;
        let arr: Vec<Value> = list
            .bind(obj.py())
            .iter()
            .map(|item| python_to_serde_value(item))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(Value::Array(arr));
    }
    Err(PyTypeError::new_err(format!(
        "unsupported type for JSON conversion: {}",
        obj.get_type().name()?
    )))
}

fn serde_to_python(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => {
            let bound = Bound::clone(&PyBool::new(py, *b));
            Ok(bound.unbind().into())
        }
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py).unwrap().unbind().into())
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_pyobject(py).unwrap().unbind().into())
            } else if let Some(f) = n.as_f64() {
                Ok(Clone::clone(&PyFloat::new(py, f)).unbind().into())
            } else {
                Ok(py.None())
            }
        }
        Value::String(s) => Ok(Clone::clone(&PyString::new(py, s)).unbind().into()),
        Value::Array(arr) => {
            let list = PyList::empty(py);
            for v in arr {
                list.append(serde_to_python(py, v)?)?;
            }
            Ok(Clone::clone(&list).unbind().into())
        }
        Value::Object(obj) => {
            let dict = PyDict::new(py);
            for (k, v) in obj {
                dict.set_item(k.as_str(), serde_to_python(py, v)?)?;
            }
            Ok(Clone::clone(&dict).unbind().into())
        }
    }
}

#[pymodule]
mod _core {
    use super::*;

    #[pyclass]
    pub struct BcsConnection {
        inner: Mutex<transport::BcsConnection>,
    }

    #[pymethods]
    impl BcsConnection {
        #[new]
        #[pyo3(signature = (addr, port, recv_timeout_ms=None, send_timeout_ms=None, use_curve=true))]
        fn new(
            addr: String,
            port: u16,
            recv_timeout_ms: Option<u64>,
            send_timeout_ms: Option<u64>,
            use_curve: bool,
        ) -> PyResult<Self> {
            let recv = Duration::from_millis(recv_timeout_ms.unwrap_or(5000));
            let send = Duration::from_millis(send_timeout_ms.unwrap_or(5000));
            let conn = transport::BcsConnection::connect(&addr, port, recv, send, use_curve)?;
            Ok(Self {
                inner: Mutex::new(conn),
            })
        }

        fn bcs_request<'py>(
            &self,
            py: Python<'py>,
            command_name: &str,
            params: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let dict = params.cast::<PyDict>().map_err(|_| {
                PyTypeError::new_err("params must be a dict")
            })?;
            let map = python_to_serde_map(dict)?;
            let guard = self.inner.lock().map_err(|e| {
                PyRuntimeError::new_err(format!("lock poisoned: {}", e))
            })?;
            let resp_val = guard.bcs_request(command_name, map)?;
            let resp_py = serde_to_python(py, &resp_val)?;
            Ok(resp_py.into_bound(py))
        }
    }

    #[pyfunction]
    fn decode_z85(data: &str) -> PyResult<Vec<u8>> {
        Ok(z85::decode_scalar(data.as_bytes())?)
    }

    #[pyfunction]
    fn decode_z85_parallel(py: Python<'_>, data: &str) -> PyResult<Vec<u8>> {
        #[allow(deprecated)]
        let result = py.allow_threads(|| z85::decode_parallel(data.as_bytes()));
        Ok(result?)
    }
}
