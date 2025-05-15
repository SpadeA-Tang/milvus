package typeutil

type Either[L any, R any] struct {
	left   L
	right  R
	isLeft bool
}

func NewLeft[L any, R any](value L) Either[L, R] {
	var zero R
	return Either[L, R]{left: value, right: zero, isLeft: true}
}

func NewRight[L any, R any](value R) Either[L, R] {
	var zero L
	return Either[L, R]{left: zero, right: value, isLeft: false}
}

func (e Either[L, R]) IsLeft() bool {
	return e.isLeft
}

func (e Either[L, R]) IsRight() bool {
	return !e.isLeft
}

func (e Either[L, R]) Left() L {
	if !e.isLeft {
		panic("Called Left on a Right value")
	}
	return e.left
}

func (e Either[L, R]) Right() R {
	if e.isLeft {
		panic("Called Right on a Left value")
	}
	return e.right
}
