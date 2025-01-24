// Code generated by mockery v2.46.0. DO NOT EDIT.

package compaction

import (
	datapb "github.com/milvus-io/milvus/pkg/proto/datapb"
	mock "github.com/stretchr/testify/mock"
)

// MockCompactor is an autogenerated mock type for the Compactor type
type MockCompactor struct {
	mock.Mock
}

type MockCompactor_Expecter struct {
	mock *mock.Mock
}

func (_m *MockCompactor) EXPECT() *MockCompactor_Expecter {
	return &MockCompactor_Expecter{mock: &_m.Mock}
}

// Compact provides a mock function with given fields:
func (_m *MockCompactor) Compact() (*datapb.CompactionPlanResult, error) {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for Compact")
	}

	var r0 *datapb.CompactionPlanResult
	var r1 error
	if rf, ok := ret.Get(0).(func() (*datapb.CompactionPlanResult, error)); ok {
		return rf()
	}
	if rf, ok := ret.Get(0).(func() *datapb.CompactionPlanResult); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*datapb.CompactionPlanResult)
		}
	}

	if rf, ok := ret.Get(1).(func() error); ok {
		r1 = rf()
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockCompactor_Compact_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Compact'
type MockCompactor_Compact_Call struct {
	*mock.Call
}

// Compact is a helper method to define mock.On call
func (_e *MockCompactor_Expecter) Compact() *MockCompactor_Compact_Call {
	return &MockCompactor_Compact_Call{Call: _e.mock.On("Compact")}
}

func (_c *MockCompactor_Compact_Call) Run(run func()) *MockCompactor_Compact_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockCompactor_Compact_Call) Return(_a0 *datapb.CompactionPlanResult, _a1 error) *MockCompactor_Compact_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockCompactor_Compact_Call) RunAndReturn(run func() (*datapb.CompactionPlanResult, error)) *MockCompactor_Compact_Call {
	_c.Call.Return(run)
	return _c
}

// Complete provides a mock function with given fields:
func (_m *MockCompactor) Complete() {
	_m.Called()
}

// MockCompactor_Complete_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Complete'
type MockCompactor_Complete_Call struct {
	*mock.Call
}

// Complete is a helper method to define mock.On call
func (_e *MockCompactor_Expecter) Complete() *MockCompactor_Complete_Call {
	return &MockCompactor_Complete_Call{Call: _e.mock.On("Complete")}
}

func (_c *MockCompactor_Complete_Call) Run(run func()) *MockCompactor_Complete_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockCompactor_Complete_Call) Return() *MockCompactor_Complete_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockCompactor_Complete_Call) RunAndReturn(run func()) *MockCompactor_Complete_Call {
	_c.Call.Return(run)
	return _c
}

// GetChannelName provides a mock function with given fields:
func (_m *MockCompactor) GetChannelName() string {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for GetChannelName")
	}

	var r0 string
	if rf, ok := ret.Get(0).(func() string); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(string)
	}

	return r0
}

// MockCompactor_GetChannelName_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetChannelName'
type MockCompactor_GetChannelName_Call struct {
	*mock.Call
}

// GetChannelName is a helper method to define mock.On call
func (_e *MockCompactor_Expecter) GetChannelName() *MockCompactor_GetChannelName_Call {
	return &MockCompactor_GetChannelName_Call{Call: _e.mock.On("GetChannelName")}
}

func (_c *MockCompactor_GetChannelName_Call) Run(run func()) *MockCompactor_GetChannelName_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockCompactor_GetChannelName_Call) Return(_a0 string) *MockCompactor_GetChannelName_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockCompactor_GetChannelName_Call) RunAndReturn(run func() string) *MockCompactor_GetChannelName_Call {
	_c.Call.Return(run)
	return _c
}

// GetCollection provides a mock function with given fields:
func (_m *MockCompactor) GetCollection() int64 {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for GetCollection")
	}

	var r0 int64
	if rf, ok := ret.Get(0).(func() int64); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(int64)
	}

	return r0
}

// MockCompactor_GetCollection_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetCollection'
type MockCompactor_GetCollection_Call struct {
	*mock.Call
}

// GetCollection is a helper method to define mock.On call
func (_e *MockCompactor_Expecter) GetCollection() *MockCompactor_GetCollection_Call {
	return &MockCompactor_GetCollection_Call{Call: _e.mock.On("GetCollection")}
}

func (_c *MockCompactor_GetCollection_Call) Run(run func()) *MockCompactor_GetCollection_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockCompactor_GetCollection_Call) Return(_a0 int64) *MockCompactor_GetCollection_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockCompactor_GetCollection_Call) RunAndReturn(run func() int64) *MockCompactor_GetCollection_Call {
	_c.Call.Return(run)
	return _c
}

// GetCompactionType provides a mock function with given fields:
func (_m *MockCompactor) GetCompactionType() datapb.CompactionType {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for GetCompactionType")
	}

	var r0 datapb.CompactionType
	if rf, ok := ret.Get(0).(func() datapb.CompactionType); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(datapb.CompactionType)
	}

	return r0
}

// MockCompactor_GetCompactionType_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetCompactionType'
type MockCompactor_GetCompactionType_Call struct {
	*mock.Call
}

// GetCompactionType is a helper method to define mock.On call
func (_e *MockCompactor_Expecter) GetCompactionType() *MockCompactor_GetCompactionType_Call {
	return &MockCompactor_GetCompactionType_Call{Call: _e.mock.On("GetCompactionType")}
}

func (_c *MockCompactor_GetCompactionType_Call) Run(run func()) *MockCompactor_GetCompactionType_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockCompactor_GetCompactionType_Call) Return(_a0 datapb.CompactionType) *MockCompactor_GetCompactionType_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockCompactor_GetCompactionType_Call) RunAndReturn(run func() datapb.CompactionType) *MockCompactor_GetCompactionType_Call {
	_c.Call.Return(run)
	return _c
}

// GetPlanID provides a mock function with given fields:
func (_m *MockCompactor) GetPlanID() int64 {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for GetPlanID")
	}

	var r0 int64
	if rf, ok := ret.Get(0).(func() int64); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(int64)
	}

	return r0
}

// MockCompactor_GetPlanID_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetPlanID'
type MockCompactor_GetPlanID_Call struct {
	*mock.Call
}

// GetPlanID is a helper method to define mock.On call
func (_e *MockCompactor_Expecter) GetPlanID() *MockCompactor_GetPlanID_Call {
	return &MockCompactor_GetPlanID_Call{Call: _e.mock.On("GetPlanID")}
}

func (_c *MockCompactor_GetPlanID_Call) Run(run func()) *MockCompactor_GetPlanID_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockCompactor_GetPlanID_Call) Return(_a0 int64) *MockCompactor_GetPlanID_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockCompactor_GetPlanID_Call) RunAndReturn(run func() int64) *MockCompactor_GetPlanID_Call {
	_c.Call.Return(run)
	return _c
}

// GetSlotUsage provides a mock function with given fields:
func (_m *MockCompactor) GetSlotUsage() int64 {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for GetSlotUsage")
	}

	var r0 int64
	if rf, ok := ret.Get(0).(func() int64); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(int64)
	}

	return r0
}

// MockCompactor_GetSlotUsage_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetSlotUsage'
type MockCompactor_GetSlotUsage_Call struct {
	*mock.Call
}

// GetSlotUsage is a helper method to define mock.On call
func (_e *MockCompactor_Expecter) GetSlotUsage() *MockCompactor_GetSlotUsage_Call {
	return &MockCompactor_GetSlotUsage_Call{Call: _e.mock.On("GetSlotUsage")}
}

func (_c *MockCompactor_GetSlotUsage_Call) Run(run func()) *MockCompactor_GetSlotUsage_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockCompactor_GetSlotUsage_Call) Return(_a0 int64) *MockCompactor_GetSlotUsage_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockCompactor_GetSlotUsage_Call) RunAndReturn(run func() int64) *MockCompactor_GetSlotUsage_Call {
	_c.Call.Return(run)
	return _c
}

// Stop provides a mock function with given fields:
func (_m *MockCompactor) Stop() {
	_m.Called()
}

// MockCompactor_Stop_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Stop'
type MockCompactor_Stop_Call struct {
	*mock.Call
}

// Stop is a helper method to define mock.On call
func (_e *MockCompactor_Expecter) Stop() *MockCompactor_Stop_Call {
	return &MockCompactor_Stop_Call{Call: _e.mock.On("Stop")}
}

func (_c *MockCompactor_Stop_Call) Run(run func()) *MockCompactor_Stop_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockCompactor_Stop_Call) Return() *MockCompactor_Stop_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockCompactor_Stop_Call) RunAndReturn(run func()) *MockCompactor_Stop_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockCompactor creates a new instance of MockCompactor. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockCompactor(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockCompactor {
	mock := &MockCompactor{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
