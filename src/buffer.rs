use crate::utils::ast::ElemSize;

use lazy_static::lazy_static;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::BTreeMap;
use strum_macros::EnumIter;

#[pyclass(eq, frozen)]
#[derive(Clone, Debug, PartialEq)]
pub enum ExtType {
    Pointer(ElemSize),
}

lazy_static! {
    static ref TYPEMAP: BTreeMap<&'static str, ElemSize> = vec![
        ("b1", ElemSize::Bool),
        ("i1", ElemSize::I8),
        ("i2", ElemSize::I16),
        ("i4", ElemSize::I32),
        ("i8", ElemSize::I64),
        ("u1", ElemSize::U8),
        ("u2", ElemSize::U16),
        ("u4", ElemSize::U32),
        ("u8", ElemSize::U64),
        ("f2", ElemSize::F16),
        ("f4", ElemSize::F32),
        ("f8", ElemSize::F64),
    ].into_iter().collect::<_>();
}

#[pymethods]
impl ElemSize {
    #[new]
    fn new(tyid: char, itemsize: char) -> PyResult<ElemSize> {
        let s = format!("{0}{1}", tyid, itemsize);
        if let Some(ty) = TYPEMAP.get(s.as_str()) {
            Ok(ty.clone())
        } else {
            Err(PyRuntimeError::new_err(format!("Unknown type identifier {s}")))
        }
    }

    fn __str__(&self) -> String {
        let s = match self {
            ElemSize::Bool => "b1",
            ElemSize::I8 => "i1",
            ElemSize::I16 => "i2",
            ElemSize::I32 => "i4",
            ElemSize::I64 => "i8",
            ElemSize::U8 => "u1",
            ElemSize::U16 => "u2",
            ElemSize::U32 => "u4",
            ElemSize::U64 => "u8",
            ElemSize::F16 => "f2",
            ElemSize::F32 => "f4",
            ElemSize::F64 => "f8",
        };
        s.to_string()
    }

    fn size(&self) -> usize {
        match self {
            ElemSize::Bool | ElemSize::I8 | ElemSize::U8 => 1,
            ElemSize::I16 | ElemSize::U16 | ElemSize::F16 => 2,
            ElemSize::I32 | ElemSize::U32 | ElemSize::F32 => 4,
            ElemSize::I64 | ElemSize::U64 | ElemSize::F64 => 8,
        }
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let builtins = py.import("builtins")?;
        let np = py.import("numpy")?;
        match self {
            ElemSize::Bool => builtins.getattr("bool"),
            ElemSize::I8 => np.getattr("int8"),
            ElemSize::I16 => np.getattr("int16"),
            ElemSize::I32 => np.getattr("int32"),
            ElemSize::I64 => np.getattr("int64"),
            ElemSize::U8 => np.getattr("uint8"),
            ElemSize::U16 => np.getattr("uint16"),
            ElemSize::U32 => np.getattr("uint32"),
            ElemSize::U64 => np.getattr("uint64"),
            ElemSize::F16 => np.getattr("float16"),
            ElemSize::F32 => np.getattr("float32"),
            ElemSize::F64 => np.getattr("float64"),
        }
    }

    // Converts the ElemSize to a dtype in PyTorch. Note how U16, U32, and U64 are translated to
    // signed integer types. We do this to work around limitations of PyTorch tensors - we treat
    // its data as unsigned internally in ParPy code.
    fn to_torch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let torch = py.import("torch")?;
        match self {
            ElemSize::Bool => torch.getattr("bool"),
            ElemSize::I8 => torch.getattr("int8"),
            ElemSize::I16 => torch.getattr("int16"),
            ElemSize::I32 => torch.getattr("int32"),
            ElemSize::I64 => torch.getattr("int64"),
            ElemSize::U8 => torch.getattr("uint8"),
            ElemSize::U16 => torch.getattr("int16"),
            ElemSize::U32 => torch.getattr("int32"),
            ElemSize::U64 => torch.getattr("int64"),
            ElemSize::F16 => torch.getattr("float16"),
            ElemSize::F32 => torch.getattr("float32"),
            ElemSize::F64 => torch.getattr("float64"),
        }
    }

    fn to_ctype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let ctypes = py.import("ctypes")?;
        match self {
            ElemSize::Bool => ctypes.getattr("c_bool"),
            ElemSize::I8 => ctypes.getattr("c_int8"),
            ElemSize::I16 => ctypes.getattr("c_int16"),
            ElemSize::I32 => ctypes.getattr("c_int32"),
            ElemSize::I64 => ctypes.getattr("c_int64"),
            ElemSize::U8 => ctypes.getattr("c_uint8"),
            ElemSize::U16 => ctypes.getattr("c_uint16"),
            ElemSize::U32 => ctypes.getattr("c_uint32"),
            ElemSize::U64 => ctypes.getattr("c_uint64"),
            ElemSize::F16 => ctypes.getattr("c_uint16"),
            ElemSize::F32 => ctypes.getattr("c_float"),
            ElemSize::F64 => ctypes.getattr("c_double"),
        }
    }
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, EnumIter)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
    NotRelevant
}

#[pymethods]
impl ByteOrder {
    #[new]
    fn new(bo: char) -> PyResult<ByteOrder> {
        match bo {
            '<' => Ok(ByteOrder::LittleEndian),
            '>' => Ok(ByteOrder::BigEndian),
            '|' => Ok(ByteOrder::NotRelevant),
            _ => Err(PyRuntimeError::new_err(format!("Unknown byteorder {bo}"))),
        }
    }

    fn __str__(&self) -> String {
        let s = match self {
            ByteOrder::LittleEndian => "<",
            ByteOrder::BigEndian => ">",
            ByteOrder::NotRelevant => "|",
        };
        s.to_string()
    }
}

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct DataType {
    pub byteorder: ByteOrder,
    pub sz: ElemSize
}

fn find_byte_order() -> ByteOrder {
    if cfg!(target_endian = "big") {
        ByteOrder::BigEndian
    } else {
        ByteOrder::LittleEndian
    }
}

#[pymethods]
impl DataType {
    #[new]
    fn new(s: &str) -> PyResult<DataType> {
        if let [byteorder, tyid, itemsize] = s.as_bytes() {
            Ok(DataType {
                byteorder: ByteOrder::new(char::from(*byteorder))?,
                sz: ElemSize::new(char::from(*tyid), char::from(*itemsize))?
            })
        } else {
            Err(PyRuntimeError::new_err(format!("Unknown format of type specification: {s}")))
        }
    }

    fn __str__(&self) -> String {
        format!("{0}{1}", self.byteorder.__str__(), self.sz.__str__())
    }

    fn __eq__(&self, other: &DataType) -> bool {
        self.byteorder == other.byteorder && self.sz == other.sz
    }

    #[staticmethod]
    fn from_elem_size(sz: ElemSize) -> DataType {
        let byteorder = find_byte_order();
        DataType {byteorder, sz}
    }

    fn size(&self) -> usize {
        self.sz.size()
    }

    fn is_signed_integer(&self) -> bool {
        match self.sz {
            ElemSize::I8 | ElemSize::I16 | ElemSize::I32 | ElemSize::I64 => true,
            _ => false
        }
    }

    fn is_unsigned_integer(&self) -> bool {
        match self.sz {
            ElemSize::U8 | ElemSize::U16 | ElemSize::U32 | ElemSize::U64 => true,
            _ => false
        }
    }

    fn is_integer(&self) -> bool {
        self.is_signed_integer() || self.is_unsigned_integer()
    }

    fn is_floating_point(&self) -> bool {
        match self.sz {
            ElemSize::F16 | ElemSize::F32 | ElemSize::F64 => true,
            _ => false
        }
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.sz.to_numpy(py)
    }

    fn to_torch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.sz.to_torch(py)
    }

    fn to_ctype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.sz.to_ctype(py)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;

    use strum::IntoEnumIterator;

    #[test]
    fn data_type_parse_bool() {
        let expected = DataType {byteorder: ByteOrder::LittleEndian, sz: ElemSize::Bool};
        assert_eq!(DataType::new("<b1").unwrap(), expected);
    }

    #[test]
    fn data_type_parse_float16() {
        let expected = DataType {byteorder: ByteOrder::BigEndian, sz: ElemSize::F16};
        assert_eq!(DataType::new(">f2").unwrap(), expected);
    }

    #[test]
    fn data_type_parse_int32() {
        let expected = DataType {byteorder: ByteOrder::NotRelevant, sz: ElemSize::I32};
        assert_eq!(DataType::new("|i4").unwrap(), expected);
    }

    #[test]
    fn data_type_parse_complex_fails() {
        assert_py_error_matches(DataType::new("<c8"), "type identifier");
    }

    #[test]
    fn data_type_parse_invalid_byte_order() {
        assert_py_error_matches(DataType::new("?b1"), "byteorder");
    }

    #[test]
    fn print_and_parse_identity_fun() {
        for sz in ElemSize::iter() {
            for byteorder in ByteOrder::iter() {
                let dt = DataType {byteorder, sz: sz.clone()};
                let dt_id = DataType::new(&dt.__str__()).unwrap();
                assert_eq!(dt, dt_id);
            }
        }
    }
}
