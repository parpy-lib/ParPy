from .parpy import ElemSize, ExtType

Bool = ElemSize.Bool
I8 = ElemSize.I8
I16 = ElemSize.I16
I32 = ElemSize.I32
I64 = ElemSize.I64
U8 = ElemSize.U8
U16 = ElemSize.U16
U32 = ElemSize.U32
U64 = ElemSize.U64
F16 = ElemSize.F16
F32 = ElemSize.F32
F64 = ElemSize.F64

def pointer(ty):
    if isinstance(ty, ElemSize):
        return ExtType.Pointer(ty)
    else:
        raise RuntimeError(f"Provided type {ty} must be an ExtType.")
