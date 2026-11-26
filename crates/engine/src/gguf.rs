//! GGUF files, memory-mapped: metadata, the tensor directory, and tensor data in place.
//!
//! The format (v2 and v3): magic `GGUF`, version, tensor count, metadata count, then the
//! metadata key/value pairs, the tensor infos (name, dims fastest first, type, offset), and the
//! tensor data from the next `general.alignment` boundary (default 32).

use crate::{Error, Result};
use memmap2::Mmap;
use std::{collections::BTreeMap, fs::File, path::Path};

/// A metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::U16(v) => v as u64,
            Value::U32(v) => v as u64,
            Value::U64(v) => v,
            Value::I8(v) if v >= 0 => v as u64,
            Value::I16(v) if v >= 0 => v as u64,
            Value::I32(v) if v >= 0 => v as u64,
            Value::I64(v) if v >= 0 => v as u64,
            Value::Bool(v) => v as u64,
            _ => return None,
        })
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            Value::I8(v) => Some(v as f64),
            Value::I16(v) => Some(v as f64),
            Value::I32(v) => Some(v as f64),
            Value::I64(v) => Some(v as f64),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// A short rendering for listings: arrays show their length and first items.
    pub fn summary(&self) -> String {
        match self {
            Value::Str(s) if s.len() > 80 => format!("{:?}…", &s[..s.floor_char_boundary(80)]),
            Value::Str(s) => format!("{s:?}"),
            Value::Array(items) => {
                let head: Vec<String> = items.iter().take(6).map(Value::summary).collect();
                let more = if items.len() > 6 { ", …" } else { "" };
                format!("[{}{more}] ({} items)", head.join(", "), items.len())
            }
            Value::F32(v) => format!("{v}"),
            Value::F64(v) => format!("{v}"),
            Value::Bool(v) => format!("{v}"),
            other => other
                .as_u64()
                .map(|v| v.to_string())
                .or_else(|| other.as_f64().map(|v| v.to_string()))
                .unwrap_or_default(),
        }
    }
}

/// A ggml tensor type, by its GGUF code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Dtype {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    I8,
    I16,
    I32,
    I64,
    F64,
    Bf16,
    Other(u32),
}

impl Dtype {
    fn from_code(code: u32) -> Dtype {
        match code {
            0 => Dtype::F32,
            1 => Dtype::F16,
            2 => Dtype::Q4_0,
            3 => Dtype::Q4_1,
            6 => Dtype::Q5_0,
            7 => Dtype::Q5_1,
            8 => Dtype::Q8_0,
            9 => Dtype::Q8_1,
            10 => Dtype::Q2K,
            11 => Dtype::Q3K,
            12 => Dtype::Q4K,
            13 => Dtype::Q5K,
            14 => Dtype::Q6K,
            15 => Dtype::Q8K,
            24 => Dtype::I8,
            25 => Dtype::I16,
            26 => Dtype::I32,
            27 => Dtype::I64,
            28 => Dtype::F64,
            30 => Dtype::Bf16,
            other => Dtype::Other(other),
        }
    }

    /// (elements per block, bytes per block), or `None` for a type this reader can't size.
    pub fn block(self) -> Option<(usize, usize)> {
        Some(match self {
            Dtype::F32 | Dtype::I32 => (1, 4),
            Dtype::F16 | Dtype::Bf16 | Dtype::I16 => (1, 2),
            Dtype::I8 => (1, 1),
            Dtype::I64 | Dtype::F64 => (1, 8),
            Dtype::Q4_0 => (32, 18),
            Dtype::Q4_1 => (32, 20),
            Dtype::Q5_0 => (32, 22),
            Dtype::Q5_1 => (32, 24),
            Dtype::Q8_0 => (32, 34),
            Dtype::Q8_1 => (32, 36),
            Dtype::Q2K => (256, 84),
            Dtype::Q3K => (256, 110),
            Dtype::Q4K => (256, 144),
            Dtype::Q5K => (256, 176),
            Dtype::Q6K => (256, 210),
            Dtype::Q8K => (256, 292),
            Dtype::Other(_) => return None,
        })
    }
}

/// One entry of the tensor directory.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions as ggml stores them: fastest-varying first (`ne[0]` is the row length).
    pub dims: Vec<u64>,
    pub dtype: Dtype,
    /// Byte offset of the data from the start of the file.
    pub offset: usize,
    pub bytes: usize,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

pub struct Gguf {
    map: Mmap,
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    /// In file order.
    pub tensors: Vec<TensorInfo>,
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|end| *end <= self.data.len())
            .ok_or_else(|| Error::Gguf(format!("truncated at byte {}", self.at)))?;
        let slice = &self.data[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.bytes(N)?.try_into().unwrap())
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn len(&mut self) -> Result<usize> {
        let n = self.u64()?;
        usize::try_from(n)
            .ok()
            .filter(|n| *n <= self.data.len())
            .ok_or_else(|| Error::Gguf(format!("implausible length {n} at byte {}", self.at)))
    }

    fn string(&mut self) -> Result<String> {
        let n = self.len()?;
        String::from_utf8(self.bytes(n)?.to_vec())
            .map_err(|_| Error::Gguf(format!("invalid UTF-8 before byte {}", self.at)))
    }

    fn value(&mut self, kind: u32) -> Result<Value> {
        Ok(match kind {
            0 => Value::U8(self.array::<1>()?[0]),
            1 => Value::I8(self.array::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.array()?)),
            3 => Value::I16(i16::from_le_bytes(self.array()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.array()?)),
            6 => Value::F32(f32::from_le_bytes(self.array()?)),
            7 => Value::Bool(self.array::<1>()?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let item = self.u32()?;
                let n = self.len()?;
                let mut items = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    items.push(self.value(item)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.array()?)),
            12 => Value::F64(f64::from_le_bytes(self.array()?)),
            other => return Err(Error::Gguf(format!("unknown metadata type {other}"))),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Gguf> {
        let file = File::open(path)?;
        // SAFETY: model files are read-only inputs; nothing in this process writes them.
        let map = unsafe { Mmap::map(&file)? };
        let mut r = Reader { data: &map, at: 0 };
        if r.bytes(4)? != b"GGUF" {
            return Err(Error::Gguf("not a GGUF file".into()));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::Gguf(format!("unsupported version {version}")));
        }
        let tensor_count = r.len()?;
        let kv_count = r.len()?;
        let mut metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = r.string()?;
            let kind = r.u32()?;
            metadata.insert(key, r.value(kind)?);
        }
        let mut raw = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            if n_dims > 8 {
                return Err(Error::Gguf(format!("{name}: {n_dims} dimensions")));
            }
            let dims = (0..n_dims).map(|_| r.u64()).collect::<Result<Vec<_>>>()?;
            let dtype = Dtype::from_code(r.u32()?);
            let offset = r.u64()?;
            raw.push((name, dims, dtype, offset));
        }
        let alignment = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(32) as usize;
        let data_start = r.at.next_multiple_of(alignment.max(1));
        let mut tensors = Vec::with_capacity(raw.len());
        for (name, dims, dtype, offset) in raw {
            let elements: u64 = dims.iter().product();
            let bytes = match dtype.block() {
                Some((per, size)) => (elements as usize).div_ceil(per) * size,
                None => 0,
            };
            let offset = data_start + offset as usize;
            if offset + bytes > map.len() {
                return Err(Error::Gguf(format!(
                    "{name}: data past the end of the file"
                )));
            }
            tensors.push(TensorInfo {
                name,
                dims,
                dtype,
                offset,
                bytes,
            });
        }
        Ok(Gguf {
            map,
            version,
            metadata,
            tensors,
        })
    }

    pub fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(Value::as_str)
    }

    pub fn metadata_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(Value::as_u64)
    }

    pub fn metadata_f64(&self, key: &str) -> Option<f64> {
        self.metadata.get(key).and_then(Value::as_f64)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// A tensor's bytes, in place in the mapped file.
    pub fn data(&self, info: &TensorInfo) -> &[u8] {
        &self.map[info.offset..info.offset + info.bytes]
    }

    /// A tensor as f32, for the types the engine reads on the CPU: F32, F16, BF16 and Q8_0.
    pub fn dequantize(&self, info: &TensorInfo) -> Result<Vec<f32>> {
        let data = self.data(info);
        Ok(match info.dtype {
            Dtype::F32 => data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            Dtype::F16 => data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::f16::from_le_bytes(*b).to_f32())
                .collect(),
            Dtype::Bf16 => data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::bf16::from_le_bytes(*b).to_f32())
                .collect(),
            Dtype::Q8_0 => {
                let mut out = Vec::with_capacity(info.elements() as usize);
                for block in data.as_chunks::<34>().0 {
                    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                    out.extend(block[2..].iter().map(|q| *q as i8 as f32 * scale));
                }
                out
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "dequantizing {other:?} ({})",
                    info.name
                )));
            }
        })
    }
}
