from .prickle import ElemSize, ExtType

I8 = ExtType.Scalar(ElemSize.I8)
I16 = ExtType.Scalar(ElemSize.I16)
I32 = ExtType.Scalar(ElemSize.I32)
I64 = ExtType.Scalar(ElemSize.I64)
U8 = ExtType.Scalar(ElemSize.U8)
U16 = ExtType.Scalar(ElemSize.U16)
U32 = ExtType.Scalar(ElemSize.U32)
U64 = ExtType.Scalar(ElemSize.U64)
F16 = ExtType.Scalar(ElemSize.F16)
F32 = ExtType.Scalar(ElemSize.F32)
F64 = ExtType.Scalar(ElemSize.F64)

def pointer(ty):
    if isinstance(ty, ExtType.Scalar):
        return ExtType.Pointer(ty._0)
    else:
        raise RuntimeError(f"Provided type {ty} must be an ExtType.")
