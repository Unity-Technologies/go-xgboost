package core

func copyUint32Slice(sli []uint32) []uint32 {
	n := make([]uint32, len(sli))
	copy(n, sli)
	return n
}

func copyFloat32Slice(sli []float32) []float32 {
	n := make([]float32, len(sli))
	copy(n, sli)
	return n
}
