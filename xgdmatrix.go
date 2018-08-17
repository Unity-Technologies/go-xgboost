package xgboost

/*
#cgo LDFLAGS: -lxgboost
#include <xgboost/c_api.h>
#include <stdio.h>
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"runtime"
	"unsafe"
)

// ErrNotSuccessful returned when an action fails
var ErrNotSuccessful = errors.New("not succesfull")

// XGDMatrix matrix
type XGDMatrix struct {
	handle C.DMatrixHandle
	cols   int
	rows   int
}

// SetUIntInfo set uint32 vector to a content in info
func (matrix *XGDMatrix) SetUIntInfo(field string, values []uint32) error {
	cstr := C.CString(field)
	defer C.free(unsafe.Pointer(cstr))

	res := C.XGDMatrixSetUIntInfo(matrix.handle, cstr, (*C.uint)(&values[0]), C.ulong(len(values)))
	if int(res) != 0 {
		return ErrNotSuccessful
	}

	return nil
}

// SetFloatInfo set float vector to a content in info
func (matrix *XGDMatrix) SetFloatInfo(field string, values []float32) error {
	cstr := C.CString(field)
	defer C.free(unsafe.Pointer(cstr))

	res := C.XGDMatrixSetFloatInfo(matrix.handle, cstr, (*C.float)(&values[0]), C.ulong(len(values)))
	if int(res) != 0 {
		return ErrNotSuccessful
	}

	return nil
}

func xdgMatrixFinalizer(mat *XGDMatrix) {
	C.XGDMatrixFree(mat.handle)
}

// XGDMatrixCreateFromMat create matrix content from dense matrix
func XGDMatrixCreateFromMat(data []float32, nrows int, ncols int, missing float32) (*XGDMatrix, error) {
	if len(data) != nrows*ncols {
		return nil, errors.New("data length doesn't match given dimensions")
	}

	var out C.DMatrixHandle
	res := C.XGDMatrixCreateFromMat((*C.float)(&data[0]), C.ulong(nrows), C.ulong(ncols), C.float(missing), &out)
	if int(res) != 0 {
		return nil, ErrNotSuccessful
	}

	matrix := &XGDMatrix{handle: out, rows: nrows, cols: ncols}
	runtime.SetFinalizer(matrix, xdgMatrixFinalizer)

	return matrix, nil
}
