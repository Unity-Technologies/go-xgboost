package xgboost

import (
	"runtime"

	"github.com/Applifier/go-xgboost/core"
)

// Matrix interface for 2D matrix
type Matrix interface {
	Data() (data []float32, rowCount, columnCount int)
}

// FloatSliceVector float32 slice backed Matrix implementation
type FloatSliceVector []float32

// Data returns float32 slice as (1, len(data)) matrix
func (fsm FloatSliceVector) Data() (data []float32, rowCount, columnCount int) {
	return fsm, 1, len(fsm)
}

// Predictor interface for xgboost predictors
type Predictor interface {
	Predict(input Matrix) ([]float32, error)
}

// NewPredictor returns a new predictor based on given model path, worker count, option mask, ntree_limit and missing value indicator
func NewPredictor(xboostSavedModelPath string, workerCount int, optionMask int, nTreeLimit uint, missingValue float32) (Predictor, error) {
	requestChan := make(chan multiBoosterRequest)
	initErrors := make(chan error)
	defer close(initErrors)

	for i := 0; i < workerCount; i++ {
		go func() {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()

			booster, err := core.XGBoosterCreate(nil)
			if err != nil {
				initErrors <- err
				return
			}

			err = booster.LoadModel(xboostSavedModelPath)
			if err != nil {
				initErrors <- err
				return
			}

			// No errors occured during init
			initErrors <- nil

			for req := range requestChan {
				data, rowCount, columnCount := req.matrix.Data()
				matrix, err := core.XGDMatrixCreateFromMat(data, rowCount, columnCount, missingValue)
				if err != nil {
					req.resultChan <- multiBoosterResponse{
						err: err,
					}
					continue
				}

				res, err := booster.Predict(matrix, optionMask, nTreeLimit)
				req.resultChan <- multiBoosterResponse{
					err:    err,
					result: res,
				}
			}
		}()

		err := <-initErrors
		if err != nil {
			return nil, err
		}
	}

	return &multiBooster{reqChan: requestChan}, nil
}

type multiBoosterRequest struct {
	matrix     Matrix
	resultChan chan multiBoosterResponse
}

type multiBoosterResponse struct {
	err    error
	result []float32
}

type multiBooster struct {
	reqChan chan multiBoosterRequest
}

func (mb *multiBooster) Predict(input Matrix) ([]float32, error) {
	resChan := make(chan multiBoosterResponse)
	mb.reqChan <- multiBoosterRequest{
		matrix:     input,
		resultChan: resChan,
	}

	result := <-resChan
	return result.result, result.err
}
