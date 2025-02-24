// Code generated by mockery v2.46.0. DO NOT EDIT.

package mock_types

import (
	context "context"

	types "github.com/milvus-io/milvus/pkg/v2/streaming/util/types"
	mock "github.com/stretchr/testify/mock"
)

// MockAssignmentRebalanceTrigger is an autogenerated mock type for the AssignmentRebalanceTrigger type
type MockAssignmentRebalanceTrigger struct {
	mock.Mock
}

type MockAssignmentRebalanceTrigger_Expecter struct {
	mock *mock.Mock
}

func (_m *MockAssignmentRebalanceTrigger) EXPECT() *MockAssignmentRebalanceTrigger_Expecter {
	return &MockAssignmentRebalanceTrigger_Expecter{mock: &_m.Mock}
}

// ReportAssignmentError provides a mock function with given fields: ctx, pchannel, err
func (_m *MockAssignmentRebalanceTrigger) ReportAssignmentError(ctx context.Context, pchannel types.PChannelInfo, err error) error {
	ret := _m.Called(ctx, pchannel, err)

	if len(ret) == 0 {
		panic("no return value specified for ReportAssignmentError")
	}

	var r0 error
	if rf, ok := ret.Get(0).(func(context.Context, types.PChannelInfo, error) error); ok {
		r0 = rf(ctx, pchannel, err)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockAssignmentRebalanceTrigger_ReportAssignmentError_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'ReportAssignmentError'
type MockAssignmentRebalanceTrigger_ReportAssignmentError_Call struct {
	*mock.Call
}

// ReportAssignmentError is a helper method to define mock.On call
//   - ctx context.Context
//   - pchannel types.PChannelInfo
//   - err error
func (_e *MockAssignmentRebalanceTrigger_Expecter) ReportAssignmentError(ctx interface{}, pchannel interface{}, err interface{}) *MockAssignmentRebalanceTrigger_ReportAssignmentError_Call {
	return &MockAssignmentRebalanceTrigger_ReportAssignmentError_Call{Call: _e.mock.On("ReportAssignmentError", ctx, pchannel, err)}
}

func (_c *MockAssignmentRebalanceTrigger_ReportAssignmentError_Call) Run(run func(ctx context.Context, pchannel types.PChannelInfo, err error)) *MockAssignmentRebalanceTrigger_ReportAssignmentError_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(types.PChannelInfo), args[2].(error))
	})
	return _c
}

func (_c *MockAssignmentRebalanceTrigger_ReportAssignmentError_Call) Return(_a0 error) *MockAssignmentRebalanceTrigger_ReportAssignmentError_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockAssignmentRebalanceTrigger_ReportAssignmentError_Call) RunAndReturn(run func(context.Context, types.PChannelInfo, error) error) *MockAssignmentRebalanceTrigger_ReportAssignmentError_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockAssignmentRebalanceTrigger creates a new instance of MockAssignmentRebalanceTrigger. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockAssignmentRebalanceTrigger(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockAssignmentRebalanceTrigger {
	mock := &MockAssignmentRebalanceTrigger{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
