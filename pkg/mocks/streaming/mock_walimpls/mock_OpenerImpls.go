// Code generated by mockery v2.46.0. DO NOT EDIT.

package mock_walimpls

import (
	context "context"

	walimpls "github.com/milvus-io/milvus/pkg/v2/streaming/walimpls"
	mock "github.com/stretchr/testify/mock"
)

// MockOpenerImpls is an autogenerated mock type for the OpenerImpls type
type MockOpenerImpls struct {
	mock.Mock
}

type MockOpenerImpls_Expecter struct {
	mock *mock.Mock
}

func (_m *MockOpenerImpls) EXPECT() *MockOpenerImpls_Expecter {
	return &MockOpenerImpls_Expecter{mock: &_m.Mock}
}

// Close provides a mock function with given fields:
func (_m *MockOpenerImpls) Close() {
	_m.Called()
}

// MockOpenerImpls_Close_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Close'
type MockOpenerImpls_Close_Call struct {
	*mock.Call
}

// Close is a helper method to define mock.On call
func (_e *MockOpenerImpls_Expecter) Close() *MockOpenerImpls_Close_Call {
	return &MockOpenerImpls_Close_Call{Call: _e.mock.On("Close")}
}

func (_c *MockOpenerImpls_Close_Call) Run(run func()) *MockOpenerImpls_Close_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockOpenerImpls_Close_Call) Return() *MockOpenerImpls_Close_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockOpenerImpls_Close_Call) RunAndReturn(run func()) *MockOpenerImpls_Close_Call {
	_c.Call.Return(run)
	return _c
}

// Open provides a mock function with given fields: ctx, opt
func (_m *MockOpenerImpls) Open(ctx context.Context, opt *walimpls.OpenOption) (walimpls.WALImpls, error) {
	ret := _m.Called(ctx, opt)

	if len(ret) == 0 {
		panic("no return value specified for Open")
	}

	var r0 walimpls.WALImpls
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, *walimpls.OpenOption) (walimpls.WALImpls, error)); ok {
		return rf(ctx, opt)
	}
	if rf, ok := ret.Get(0).(func(context.Context, *walimpls.OpenOption) walimpls.WALImpls); ok {
		r0 = rf(ctx, opt)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(walimpls.WALImpls)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, *walimpls.OpenOption) error); ok {
		r1 = rf(ctx, opt)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockOpenerImpls_Open_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Open'
type MockOpenerImpls_Open_Call struct {
	*mock.Call
}

// Open is a helper method to define mock.On call
//   - ctx context.Context
//   - opt *walimpls.OpenOption
func (_e *MockOpenerImpls_Expecter) Open(ctx interface{}, opt interface{}) *MockOpenerImpls_Open_Call {
	return &MockOpenerImpls_Open_Call{Call: _e.mock.On("Open", ctx, opt)}
}

func (_c *MockOpenerImpls_Open_Call) Run(run func(ctx context.Context, opt *walimpls.OpenOption)) *MockOpenerImpls_Open_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*walimpls.OpenOption))
	})
	return _c
}

func (_c *MockOpenerImpls_Open_Call) Return(_a0 walimpls.WALImpls, _a1 error) *MockOpenerImpls_Open_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockOpenerImpls_Open_Call) RunAndReturn(run func(context.Context, *walimpls.OpenOption) (walimpls.WALImpls, error)) *MockOpenerImpls_Open_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockOpenerImpls creates a new instance of MockOpenerImpls. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockOpenerImpls(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockOpenerImpls {
	mock := &MockOpenerImpls{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
