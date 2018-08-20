package core

/*
#cgo LDFLAGS: -lxgboost
#include <xgboost/c_api.h>
#include <stdio.h>
#include <stdlib.h>
*/
import "C"
import (
	"reflect"
	"runtime"
	"unsafe"
)

// XGBooster gradient booster
type XGBooster struct {
	handle C.BoosterHandle
}

// SetParam set parameters
func (booster *XGBooster) SetParam(name string, value string) error {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	cvalue := C.CString(value)
	defer C.free(unsafe.Pointer(cvalue))

	res := C.XGBoosterSetParam(booster.handle, cname, cvalue)
	if err := checkError(res); err != nil {
		return err
	}

	return nil
}

// DeleteParam set parameters
func (booster *XGBooster) DeleteParam(name string) error {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	res := C.XGBoosterSetParam(booster.handle, cname, nil)
	if err := checkError(res); err != nil {
		return err
	}

	return nil
}

// UpdateOneIter update the model in one round using dtrain
func (booster *XGBooster) UpdateOneIter(iter int, mat *XGDMatrix) error {
	res := C.XGBoosterUpdateOneIter(booster.handle, C.int(iter), mat.handle)
	if err := checkError(res); err != nil {
		return err
	}

	return nil
}

// Predict make prediction based on dmat
func (booster *XGBooster) Predict(mat *XGDMatrix, optionMask int, ntreeLimit uint) ([]float32, error) {
	var outLen C.ulong
	var outResult *C.float

	res := C.XGBoosterPredict(booster.handle, mat.handle, C.int(optionMask), C.uint(ntreeLimit), &outLen, &outResult)
	if err := checkError(res); err != nil {
		return nil, err
	}

	var list []float32
	sliceHeader := (*reflect.SliceHeader)((unsafe.Pointer(&list)))
	sliceHeader.Cap = int(outLen)
	sliceHeader.Len = int(outLen)
	sliceHeader.Data = uintptr(unsafe.Pointer(outResult))

	runtime.KeepAlive(mat)

	return copyFloat32Slice(list), nil
}

// LoadModel load model from existing file
func (booster *XGBooster) LoadModel(filePath string) error {
	cfilePath := C.CString(filePath)
	defer C.free(unsafe.Pointer(cfilePath))

	return checkError(C.XGBoosterLoadModel(booster.handle, cfilePath))
}

// SaveModel save model into file
func (booster *XGBooster) SaveModel(filePath string) error {
	cfilePath := C.CString(filePath)
	defer C.free(unsafe.Pointer(cfilePath))

	return checkError(C.XGBoosterSaveModel(booster.handle, cfilePath))
}

func xdgBoosterFinalizer(booster *XGBooster) {
	C.XGBoosterFree(booster.handle)
}

// XGBoosterCreate creates a new booster for a given matrixes
func XGBoosterCreate(matrix []*XGDMatrix) (*XGBooster, error) {
	var ptr *C.DMatrixHandle
	handles := make([]C.DMatrixHandle, len(matrix))
	for i, matrix := range matrix {
		handles[i] = matrix.handle
	}
	if len(handles) > 0 {
		ptr = (*C.DMatrixHandle)(&handles[0])
	}

	var out C.BoosterHandle
	res := C.XGBoosterCreate(ptr, C.ulong(len(handles)), &out)
	if err := checkError(res); err != nil {
		return nil, err
	}

	booster := &XGBooster{handle: out}
	runtime.SetFinalizer(booster, xdgBoosterFinalizer)

	return booster, nil
}
