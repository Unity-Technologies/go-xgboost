package xgboost_test

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"
	"path"
	"testing"

	"github.com/Applifier/go-xgboost"
)

func TestXGBoost(t *testing.T) {
	// create the training data
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

	matrix, err := xgboost.XGDMatrixCreateFromMat(trainData, rows, cols, -1)
	if err != nil {
		t.Error(err)
	}

	err = matrix.SetFloatInfo("label", trainLabels)
	if err != nil {
		t.Error(err)
	}

	booster, err := xgboost.XGBoosterCreate([]*xgboost.XGDMatrix{matrix})
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

	testData := make([]float32, cols*rows)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			testData[(i*cols)+j] = float32((i + 1) * (j + 1))
		}
	}

	testmat, err := xgboost.XGDMatrixCreateFromMat(testData, rows, cols, -1)
	if err != nil {
		t.Error(err)
	}

	res, err := booster.Predict(testmat, 0, 0)
	if err != nil {
		t.Error(err)
	}

	// TODO measure actual accuracy
	totalDiff := 0.0
	for i, label := range trainLabels {
		diff := math.Abs(float64(label - res[i]))
		totalDiff += diff
	}

	if totalDiff > 6.0 {
		t.Error("error is too large")
	}

	dir, err := ioutil.TempDir("", "go-xgboost")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir) // clean up

	savePath := path.Join(dir, "testmodel.bst")

	noErr(booster.SaveModel(savePath))

	newBooster, err := xgboost.XGBoosterCreate(nil)
	if err != nil {
		t.Error(err)
	}

	noErr(newBooster.LoadModel(savePath))

	testmat2, err := xgboost.XGDMatrixCreateFromMat(testData, rows, cols, -1)
	if err != nil {
		t.Error(err)
	}

	res, err = newBooster.Predict(testmat2, 0, 0)
	if err != nil {
		t.Error(err)
	}

	// TODO measure actual accuracy
	totalDiff = 0.0
	for i, label := range trainLabels {
		diff := math.Abs(float64(label - res[i]))
		totalDiff += diff
	}

	if totalDiff > 6.0 {
		t.Error("error is too large")
	}
}

func ExampleXGBoost() {
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

	// Create XGDMatrix for training data
	matrix, _ := xgboost.XGDMatrixCreateFromMat(trainData, rows, cols, -1)

	// Set training labels
	matrix.SetFloatInfo("label", trainLabels)

	// Create booster
	booster, _ := xgboost.XGBoosterCreate([]*xgboost.XGDMatrix{matrix})

	// Set booster parameters
	booster.SetParam("booster", "gbtree")
	booster.SetParam("objective", "reg:linear")
	booster.SetParam("max_depth", "5")
	booster.SetParam("eta", "0.1")
	booster.SetParam("min_child_weight", "1")
	booster.SetParam("subsample", "0.5")
	booster.SetParam("colsample_bytree", "1")
	booster.SetParam("num_parallel_tree", "1")

	// perform 200 learning iterations
	for iter := 0; iter < 200; iter++ {
		booster.UpdateOneIter(iter, matrix)
	}

	testData := make([]float32, cols*rows)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			testData[(i*cols)+j] = float32((i + 1) * (j + 1))
		}
	}

	// Create XGDMatrix for test data
	testmat, _ := xgboost.XGDMatrixCreateFromMat(testData, rows, cols, -1)

	// Predict
	res, _ := booster.Predict(testmat, 0, 0)

	fmt.Printf("%+v\n", res)
	// output: [1.08002 2.5686886 7.86032 29.923136 63.76062]
}
