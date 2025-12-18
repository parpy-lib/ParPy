from .parpy import ElemSize

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

def buffer(sz, shape):
    from .parpy import ExtType
    if isinstance(sz, ElemSize):
        return ExtType.Buffer(sz, shape)
    else:
        return ExtType.VarBuffer(sz, shape)

def shape_var():
    from .parpy import ShapeVar
    return ShapeVar()

def type_var():
    from .parpy import TypeVar
    return TypeVar()
