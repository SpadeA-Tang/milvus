#[allow(dead_code)]
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum TantivyDataType {
    Text,
    Keyword,
    // U64,
    I64,
    F64,
    Bool,
    JSON,
}
