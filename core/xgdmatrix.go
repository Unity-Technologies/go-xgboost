package core

/*
#cgo LDFLAGS: -lxgboost
#include <xgboost/c_api.h>
#include <stdio.h>
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"reflect"
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

// NumRow get number of rows.
func (matrix *XGDMatrix) NumRow() (uint32, error) {
	var count C.ulong
	if err := checkError(C.XGDMatrixNumRow(matrix.handle, &count)); err != nil {
		return 0, err
	}

	return uint32(count), nil
}

// NumCol get number of cols.
func (matrix *XGDMatrix) NumCol() (uint32, error) {
	var count C.ulong
	if err := checkError(C.XGDMatrixNumCol(matrix.handle, &count)); err != nil {
		return 0, err
	}

	return uint32(count), nil
}

// SetUIntInfo set uint32 vector to a content in info
func (matrix *XGDMatrix) SetUIntInfo(field string, values []uint32) error {
	cstr := C.CString(field)
	defer C.free(unsafe.Pointer(cstr))

	res := C.XGDMatrixSetUIntInfo(matrix.handle, cstr, (*C.uint)(&values[0]), C.ulong(len(values)))

	runtime.KeepAlive(values)
	return checkError(res)
}

// SetGroup set label of the training matrix
func (matrix *XGDMatrix) SetGroup(group ...uint32) error {
	res := C.XGDMatrixSetGroup(matrix.handle, (*C.uint)(&group[0]), C.ulong(len(group)))
	runtime.KeepAlive(group)
	return checkError(res)
}

// SetFloatInfo set float vector to a content in info
func (matrix *XGDMatrix) SetFloatInfo(field string, values []float32) error {
	cstr := C.CString(field)
	defer C.free(unsafe.Pointer(cstr))

	res := C.XGDMatrixSetFloatInfo(matrix.handle, cstr, (*C.float)(&values[0]), C.ulong(len(values)))
	if err := checkError(res); err != nil {
		return err
	}
	runtime.KeepAlive(values)

	return nil
}

// GetFloatInfo get float info vector from matrix
func (matrix *XGDMatrix) GetFloatInfo(field string) ([]float32, error) {
	cstr := C.CString(field)
	defer C.free(unsafe.Pointer(cstr))

	var outLen C.ulong
	var outResult *C.float

	if err := checkError(C.XGDMatrixGetFloatInfo(matrix.handle, cstr, &outLen, &outResult)); err != nil {
		return nil, err
	}

	var list []float32
	sliceHeader := (*reflect.SliceHeader)((unsafe.Pointer(&list)))
	sliceHeader.Cap = int(outLen)
	sliceHeader.Len = int(outLen)
	sliceHeader.Data = uintptr(unsafe.Pointer(outResult))

	return copyFloat32Slice(list), nil
}

// GetUIntInfo get uint32 info vector from matrix
func (matrix *XGDMatrix) GetUIntInfo(field string) ([]uint32, error) {
	cstr := C.CString(field)
	defer C.free(unsafe.Pointer(cstr))

	var outLen C.ulong
	var outResult *C.uint

	if err := checkError(C.XGDMatrixGetUIntInfo(matrix.handle, cstr, &outLen, &outResult)); err != nil {
		return nil, err
	}

	var list []uint32
	sliceHeader := (*reflect.SliceHeader)((unsafe.Pointer(&list)))
	sliceHeader.Cap = int(outLen)
	sliceHeader.Len = int(outLen)
	sliceHeader.Data = uintptr(unsafe.Pointer(outResult))

	return copyUint32Slice(list), nil
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
	if err := checkError(res); err != nil {
		return nil, err
	}

	matrix := &XGDMatrix{handle: out, rows: nrows, cols: ncols}
	runtime.SetFinalizer(matrix, xdgMatrixFinalizer)

	runtime.KeepAlive(data)

	return matrix, nil
}
