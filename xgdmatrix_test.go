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

	fmt.Printf("%+v\n", data)
}
