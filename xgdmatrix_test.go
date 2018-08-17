package xgboost

import (
	"fmt"
	"testing"
)

func TestXGDMatrix(t *testing.T) {
	data := []float32{1, 2, 3, 4}

	matrix, err := XGDMatrixCreateFromMat(data, 2, 2, -1)
	if err != nil {
		t.Error(err)
	}

	if matrix == nil {
		t.Error("matrix was not created")
	}

	err = matrix.SetFloatInfo("label", []float32{1, 2})
	if err != nil {
		t.Error(err)
	}

	vals, err := matrix.GetFloatInfo("label")
	if err != nil {
		t.Error(err)
	}

	if vals[0] != 1 || vals[1] != 2 {
		t.Error("Wrong values returned")
	}

	rowCount, err := matrix.NumRow()
	if err != nil {
		t.Error(err)
	}

	if rowCount != 2 {
		t.Error("Wrong row count returned")
	}

	colCount, err := matrix.NumCol()
	if err != nil {
		t.Error(err)
	}

	if colCount != 2 {
		t.Error("Wrong col count returned")
	}

	fmt.Printf("%+v\n", data)
}
