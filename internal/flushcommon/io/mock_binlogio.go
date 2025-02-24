// Code generated by mockery v2.46.0. DO NOT EDIT.

package io

import (
	context "context"

	conc "github.com/milvus-io/milvus/pkg/v2/util/conc"

	mock "github.com/stretchr/testify/mock"
)

// MockBinlogIO is an autogenerated mock type for the BinlogIO type
type MockBinlogIO struct {
	mock.Mock
}

type MockBinlogIO_Expecter struct {
	mock *mock.Mock
}

func (_m *MockBinlogIO) EXPECT() *MockBinlogIO_Expecter {
	return &MockBinlogIO_Expecter{mock: &_m.Mock}
}

// AsyncDownload provides a mock function with given fields: ctx, paths
func (_m *MockBinlogIO) AsyncDownload(ctx context.Context, paths []string) []*conc.Future[interface{}] {
	ret := _m.Called(ctx, paths)

	if len(ret) == 0 {
		panic("no return value specified for AsyncDownload")
	}

	var r0 []*conc.Future[interface{}]
	if rf, ok := ret.Get(0).(func(context.Context, []string) []*conc.Future[interface{}]); ok {
		r0 = rf(ctx, paths)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]*conc.Future[interface{}])
		}
	}

	return r0
}

// MockBinlogIO_AsyncDownload_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'AsyncDownload'
type MockBinlogIO_AsyncDownload_Call struct {
	*mock.Call
}

// AsyncDownload is a helper method to define mock.On call
//   - ctx context.Context
//   - paths []string
func (_e *MockBinlogIO_Expecter) AsyncDownload(ctx interface{}, paths interface{}) *MockBinlogIO_AsyncDownload_Call {
	return &MockBinlogIO_AsyncDownload_Call{Call: _e.mock.On("AsyncDownload", ctx, paths)}
}

func (_c *MockBinlogIO_AsyncDownload_Call) Run(run func(ctx context.Context, paths []string)) *MockBinlogIO_AsyncDownload_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].([]string))
	})
	return _c
}

func (_c *MockBinlogIO_AsyncDownload_Call) Return(_a0 []*conc.Future[interface{}]) *MockBinlogIO_AsyncDownload_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockBinlogIO_AsyncDownload_Call) RunAndReturn(run func(context.Context, []string) []*conc.Future[interface{}]) *MockBinlogIO_AsyncDownload_Call {
	_c.Call.Return(run)
	return _c
}

// AsyncUpload provides a mock function with given fields: ctx, kvs
func (_m *MockBinlogIO) AsyncUpload(ctx context.Context, kvs map[string][]byte) []*conc.Future[interface{}] {
	ret := _m.Called(ctx, kvs)

	if len(ret) == 0 {
		panic("no return value specified for AsyncUpload")
	}

	var r0 []*conc.Future[interface{}]
	if rf, ok := ret.Get(0).(func(context.Context, map[string][]byte) []*conc.Future[interface{}]); ok {
		r0 = rf(ctx, kvs)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]*conc.Future[interface{}])
		}
	}

	return r0
}

// MockBinlogIO_AsyncUpload_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'AsyncUpload'
type MockBinlogIO_AsyncUpload_Call struct {
	*mock.Call
}

// AsyncUpload is a helper method to define mock.On call
//   - ctx context.Context
//   - kvs map[string][]byte
func (_e *MockBinlogIO_Expecter) AsyncUpload(ctx interface{}, kvs interface{}) *MockBinlogIO_AsyncUpload_Call {
	return &MockBinlogIO_AsyncUpload_Call{Call: _e.mock.On("AsyncUpload", ctx, kvs)}
}

func (_c *MockBinlogIO_AsyncUpload_Call) Run(run func(ctx context.Context, kvs map[string][]byte)) *MockBinlogIO_AsyncUpload_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(map[string][]byte))
	})
	return _c
}

func (_c *MockBinlogIO_AsyncUpload_Call) Return(_a0 []*conc.Future[interface{}]) *MockBinlogIO_AsyncUpload_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockBinlogIO_AsyncUpload_Call) RunAndReturn(run func(context.Context, map[string][]byte) []*conc.Future[interface{}]) *MockBinlogIO_AsyncUpload_Call {
	_c.Call.Return(run)
	return _c
}

// Download provides a mock function with given fields: ctx, paths
func (_m *MockBinlogIO) Download(ctx context.Context, paths []string) ([][]byte, error) {
	ret := _m.Called(ctx, paths)

	if len(ret) == 0 {
		panic("no return value specified for Download")
	}

	var r0 [][]byte
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, []string) ([][]byte, error)); ok {
		return rf(ctx, paths)
	}
	if rf, ok := ret.Get(0).(func(context.Context, []string) [][]byte); ok {
		r0 = rf(ctx, paths)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([][]byte)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, []string) error); ok {
		r1 = rf(ctx, paths)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockBinlogIO_Download_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Download'
type MockBinlogIO_Download_Call struct {
	*mock.Call
}

// Download is a helper method to define mock.On call
//   - ctx context.Context
//   - paths []string
func (_e *MockBinlogIO_Expecter) Download(ctx interface{}, paths interface{}) *MockBinlogIO_Download_Call {
	return &MockBinlogIO_Download_Call{Call: _e.mock.On("Download", ctx, paths)}
}

func (_c *MockBinlogIO_Download_Call) Run(run func(ctx context.Context, paths []string)) *MockBinlogIO_Download_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].([]string))
	})
	return _c
}

func (_c *MockBinlogIO_Download_Call) Return(_a0 [][]byte, _a1 error) *MockBinlogIO_Download_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockBinlogIO_Download_Call) RunAndReturn(run func(context.Context, []string) ([][]byte, error)) *MockBinlogIO_Download_Call {
	_c.Call.Return(run)
	return _c
}

// Upload provides a mock function with given fields: ctx, kvs
func (_m *MockBinlogIO) Upload(ctx context.Context, kvs map[string][]byte) error {
	ret := _m.Called(ctx, kvs)

	if len(ret) == 0 {
		panic("no return value specified for Upload")
	}

	var r0 error
	if rf, ok := ret.Get(0).(func(context.Context, map[string][]byte) error); ok {
		r0 = rf(ctx, kvs)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockBinlogIO_Upload_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Upload'
type MockBinlogIO_Upload_Call struct {
	*mock.Call
}

// Upload is a helper method to define mock.On call
//   - ctx context.Context
//   - kvs map[string][]byte
func (_e *MockBinlogIO_Expecter) Upload(ctx interface{}, kvs interface{}) *MockBinlogIO_Upload_Call {
	return &MockBinlogIO_Upload_Call{Call: _e.mock.On("Upload", ctx, kvs)}
}

func (_c *MockBinlogIO_Upload_Call) Run(run func(ctx context.Context, kvs map[string][]byte)) *MockBinlogIO_Upload_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(map[string][]byte))
	})
	return _c
}

func (_c *MockBinlogIO_Upload_Call) Return(_a0 error) *MockBinlogIO_Upload_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockBinlogIO_Upload_Call) RunAndReturn(run func(context.Context, map[string][]byte) error) *MockBinlogIO_Upload_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockBinlogIO creates a new instance of MockBinlogIO. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockBinlogIO(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockBinlogIO {
	mock := &MockBinlogIO{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
