package xgboost

import (
	"testing"
)

func TestXGBoost(t *testing.T) {
	// create the train data
	cols := 3
	rows := 5
	trainData := make([]float32, cols*rows)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			trainData[(i*cols)+j] = float32((i + 1) * (j + 1))
		}
	}

	trainLabels := make([]float32, rows)
	for i := 0; i < rows; i++ {
		trainLabels[i] = float32(1 + i*i*i)
	}

	matrix, err := XGDMatrixCreateFromMat(trainData, rows, cols, -1)
	if err != nil {
		t.Error(err)
	}

	err = matrix.SetFloatInfo("label", trainLabels)
	if err != nil {
		t.Error(err)
	}

	booster, err := XGBoosterCreate([]*XGDMatrix{matrix})
	if err != nil {
		t.Error(err)
	}

	noErr := func(err error) {
		if err != nil {
			t.Error(err)
		}
	}

	noErr(booster.SetParam("booster", "gbtree"))
	noErr(booster.SetParam("objective", "reg:linear"))
	noErr(booster.SetParam("max_depth", "5"))
	noErr(booster.SetParam("eta", "0.1"))
	noErr(booster.SetParam("min_child_weight", "1"))
	noErr(booster.SetParam("subsample", "0.5"))
	noErr(booster.SetParam("colsample_bytree", "1"))
	noErr(booster.SetParam("num_parallel_tree", "1"))

	// perform 200 learning iterations
	for iter := 0; iter < 200; iter++ {
		noErr(booster.UpdateOneIter(iter, matrix))
	}

	testrows := 7
	testData := make([]float32, cols*testrows)
	for i := 0; i < testrows; i++ {
		for j := 0; j < cols; j++ {
			testData[(i*cols)+j] = float32((i + 1) * (j + 1))
		}
	}

	testmat, err := XGDMatrixCreateFromMat(testData, testrows, cols, -1)
	if err != nil {
		t.Error(err)
	}

	res, err := booster.Predict(testmat, 0, 0)
	if err != nil {
		t.Error(err)
	}

	for i := 0; i < len(res)-1; i++ {
		if res[i] > res[i+1] {
			t.Error("results should be ascending")
		}
	}
}
