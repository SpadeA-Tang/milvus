// Code generated by mockery v2.46.0. DO NOT EDIT.

package mock_message

import (
	message "github.com/milvus-io/milvus/pkg/v2/streaming/util/message"
	mock "github.com/stretchr/testify/mock"
)

// MockImmutableMessage is an autogenerated mock type for the ImmutableMessage type
type MockImmutableMessage struct {
	mock.Mock
}

type MockImmutableMessage_Expecter struct {
	mock *mock.Mock
}

func (_m *MockImmutableMessage) EXPECT() *MockImmutableMessage_Expecter {
	return &MockImmutableMessage_Expecter{mock: &_m.Mock}
}

// BarrierTimeTick provides a mock function with given fields:
func (_m *MockImmutableMessage) BarrierTimeTick() uint64 {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for BarrierTimeTick")
	}

	var r0 uint64
	if rf, ok := ret.Get(0).(func() uint64); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(uint64)
	}

	return r0
}

// MockImmutableMessage_BarrierTimeTick_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'BarrierTimeTick'
type MockImmutableMessage_BarrierTimeTick_Call struct {
	*mock.Call
}

// BarrierTimeTick is a helper method to define mock.On call
func (_e *MockImmutableMessage_Expecter) BarrierTimeTick() *MockImmutableMessage_BarrierTimeTick_Call {
	return &MockImmutableMessage_BarrierTimeTick_Call{Call: _e.mock.On("BarrierTimeTick")}
}

func (_c *MockImmutableMessage_BarrierTimeTick_Call) Run(run func()) *MockImmutableMessage_BarrierTimeTick_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockImmutableMessage_BarrierTimeTick_Call) Return(_a0 uint64) *MockImmutableMessage_BarrierTimeTick_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockImmutableMessage_BarrierTimeTick_Call) RunAndReturn(run func() uint64) *MockImmutableMessage_BarrierTimeTick_Call {
	_c.Call.Return(run)
	return _c
}

// BroadcastHeader provides a mock function with given fields:
func (_m *MockImmutableMessage) BroadcastHeader() *message.BroadcastHeader {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for BroadcastHeader")
	}

	var r0 *message.BroadcastHeader
	if rf, ok := ret.Get(0).(func() *message.BroadcastHeader); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*message.BroadcastHeader)
		}
	}

	return r0
}

// MockImmutableMessage_BroadcastHeader_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'BroadcastHeader'
type MockImmutableMessage_BroadcastHeader_Call struct {
	*mock.Call
}

// BroadcastHeader is a helper method to define mock.On call
func (_e *MockImmutableMessage_Expecter) BroadcastHeader() *MockImmutableMessage_BroadcastHeader_Call {
	return &MockImmutableMessage_BroadcastHeader_Call{Call: _e.mock.On("BroadcastHeader")}
}

func (_c *MockImmutableMessage_BroadcastHeader_Call) Run(run func()) *MockImmutableMessage_BroadcastHeader_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockImmutableMessage_BroadcastHeader_Call) Return(_a0 *message.BroadcastHeader) *MockImmutableMessage_BroadcastHeader_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockImmutableMessage_BroadcastHeader_Call) RunAndReturn(run func() *message.BroadcastHeader) *MockImmutableMessage_BroadcastHeader_Call {
	_c.Call.Return(run)
	return _c
}

// EstimateSize provides a mock function with given fields:
func (_m *MockImmutableMessage) EstimateSize() int {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for EstimateSize")
	}

	var r0 int
	if rf, ok := ret.Get(0).(func() int); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(int)
	}

	return r0
}

// MockImmutableMessage_EstimateSize_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'EstimateSize'
type MockImmutableMessage_EstimateSize_Call struct {
	*mock.Call
}

// EstimateSize is a helper method to define mock.On call
func (_e *MockImmutableMessage_Expecter) EstimateSize() *MockImmutableMessage_EstimateSize_Call {
	return &MockImmutableMessage_EstimateSize_Call{Call: _e.mock.On("EstimateSize")}
}

func (_c *MockImmutableMessage_EstimateSize_Call) Run(run func()) *MockImmutableMessage_EstimateSize_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockImmutableMessage_EstimateSize_Call) Return(_a0 int) *MockImmutableMessage_EstimateSize_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockImmutableMessage_EstimateSize_Call) RunAndReturn(run func() int) *MockImmutableMessage_EstimateSize_Call {
	_c.Call.Return(run)
	return _c
}

// LastConfirmedMessageID provides a mock function with given fields:
func (_m *MockImmutableMessage) LastConfirmedMessageID() message.MessageID {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for LastConfirmedMessageID")
	}

	var r0 message.MessageID
	if rf, ok := ret.Get(0).(func() message.MessageID); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(message.MessageID)
		}
	}

	return r0
}

// MockImmutableMessage_LastConfirmedMessageID_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'LastConfirmedMessageID'
type MockImmutableMessage_LastConfirmedMessageID_Call struct {
	*mock.Call
}

// LastConfirmedMessageID is a helper method to define mock.On call
func (_e *MockImmutableMessage_Expecter) LastConfirmedMessageID() *MockImmutableMessage_LastConfirmedMessageID_Call {
	return &MockImmutableMessage_LastConfirmedMessageID_Call{Call: _e.mock.On("LastConfirmedMessageID")}
}

func (_c *MockImmutableMessage_LastConfirmedMessageID_Call) Run(run func()) *MockImmutableMessage_LastConfirmedMessageID_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockImmutableMessage_LastConfirmedMessageID_Call) Return(_a0 message.MessageID) *MockImmutableMessage_LastConfirmedMessageID_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockImmutableMessage_LastConfirmedMessageID_Call) RunAndReturn(run func() message.MessageID) *MockImmutableMessage_LastConfirmedMessageID_Call {
	_c.Call.Return(run)
	return _c
}

// MessageID provides a mock function with given fields:
func (_m *MockImmutableMessage) MessageID() message.MessageID {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for MessageID")
	}

	var r0 message.MessageID
	if rf, ok := ret.Get(0).(func() message.MessageID); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(message.MessageID)
		}
	}

	return r0
}

// MockImmutableMessage_MessageID_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'MessageID'
type MockImmutableMessage_MessageID_Call struct {
	*mock.Call
}

// MessageID is a helper method to define mock.On call
func (_e *MockImmutableMessage_Expecter) MessageID() *MockImmutableMessage_MessageID_Call {
	return &MockImmutableMessage_MessageID_Call{Call: _e.mock.On("MessageID")}
}

func (_c *MockImmutableMessage_MessageID_Call) Run(run func()) *MockImmutableMessage_MessageID_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockImmutableMessage_MessageID_Call) Return(_a0 message.MessageID) *MockImmutableMessage_MessageID_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockImmutableMessage_MessageID_Call) RunAndReturn(run func() message.MessageID) *MockImmutableMessage_MessageID_Call {
	_c.Call.Return(run)
	return _c
}

// MessageType provides a mock function with given fields:
func (_m *MockImmutableMessage) MessageType() message.MessageType {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for MessageType")
	}

	var r0 message.MessageType
	if rf, ok := ret.Get(0).(func() message.MessageType); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(message.MessageType)
	}

	return r0
}

// MockImmutableMessage_MessageType_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'MessageType'
type MockImmutableMessage_MessageType_Call struct {
	*mock.Call
}

// MessageType is a helper method to define mock.On call
func (_e *MockImmutableMessage_Expecter) MessageType() *MockImmutableMessage_MessageType_Call {
	return &MockImmutableMessage_MessageType_Call{Call: _e.mock.On("MessageType")}
}

func (_c *MockImmutableMessage_MessageType_Call) Run(run func()) *MockImmutableMessage_MessageType_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockImmutableMessage_MessageType_Call) Return(_a0 message.MessageType) *MockImmutableMessage_MessageType_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockImmutableMessage_MessageType_Call) RunAndReturn(run func() message.MessageType) *MockImmutableMessage_MessageType_Call {
	_c.Call.Return(run)
	return _c
}

// Payload provides a mock function with given fields:
func (_m *MockImmutableMessage) Payload() []byte {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for Payload")
	}

	var r0 []byte
	if rf, ok := ret.Get(0).(func() []byte); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]byte)
		}
	}

	return r0
}

// MockImmutableMessage_Payload_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Payload'
type MockImmutableMessage_Payload_Call struct {
	*mock.Call
}

// Payload is a helper method to define mock.On call
func (_e *MockImmutableMessage_Expecter) Payload() *MockImmutableMessage_Payload_Call {
	return &MockImmutableMessage_Payload_Call{Call: _e.mock.On("Payload")}
}

func (_c *MockImmutableMessage_Payload_Call) Run(run func()) *MockImmutableMessage_Payload_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockImmutableMessage_Payload_Call) Return(_a0 []byte) *MockImmutableMessage_Payload_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockImmutableMessage_Payload_Call) RunAndReturn(run func() []byte) *MockImmutableMessage_Payload_Call {
	_c.Call.Return(run)
	return _c
}

// Properties provides a mock function with given fields:
func (_m *MockImmutableMessage) Properties() message.RProperties {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for Properties")
	}

	var r0 message.RProperties
	if rf, ok := ret.Get(0).(func() message.RProperties); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(message.RProperties)
		}
	}

	return r0
}

// MockImmutableMessage_Properties_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Properties'
type MockImmutableMessage_Properties_Call struct {
	*mock.Call
}

// Properties is a helper method to define mock.On call
func (_e *MockImmutableMessage_Expecter) Properties() *MockImmutableMessage_Properties_Call {
	return &MockImmutableMessage_Properties_Call{Call: _e.mock.On("Properties")}
}

func (_c *MockImmutableMessage_Properties_Call) Run(run func()) *MockImmutableMessage_Properties_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockImmutableMessage_Properties_Call) Return(_a0 message.RProperties) *MockImmutableMessage_Properties_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockImmutableMessage_Properties_Call) RunAndReturn(run func() message.RProperties) *MockImmutableMessage_Properties_Call {
	_c.Call.Return(run)
	return _c
}

// TimeTick provides a mock function with given fields:
func (_m *MockImmutableMessage) TimeTick() uint64 {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for TimeTick")
	}

	var r0 uint64
	if rf, ok := ret.Get(0).(func() uint64); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(uint64)
	}

	return r0
}

// MockImmutableMessage_TimeTick_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'TimeTick'
type MockImmutableMessage_TimeTick_Call struct {
	*mock.Call
}

// TimeTick is a helper method to define mock.On call
func (_e *MockImmutableMessage_Expecter) TimeTick() *MockImmutableMessage_TimeTick_Call {
	return &MockImmutableMessage_TimeTick_Call{Call: _e.mock.On("TimeTick")}
}

func (_c *MockImmutableMessage_TimeTick_Call) Run(run func()) *MockImmutableMessage_TimeTick_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockImmutableMessage_TimeTick_Call) Return(_a0 uint64) *MockImmutableMessage_TimeTick_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockImmutableMessage_TimeTick_Call) RunAndReturn(run func() uint64) *MockImmutableMessage_TimeTick_Call {
	_c.Call.Return(run)
	return _c
}

// TxnContext provides a mock function with given fields:
func (_m *MockImmutableMessage) TxnContext() *message.TxnContext {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for TxnContext")
	}

	var r0 *message.TxnContext
	if rf, ok := ret.Get(0).(func() *message.TxnContext); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*message.TxnContext)
		}
	}

	return r0
}

// MockImmutableMessage_TxnContext_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'TxnContext'
type MockImmutableMessage_TxnContext_Call struct {
	*mock.Call
}

// TxnContext is a helper method to define mock.On call
func (_e *MockImmutableMessage_Expecter) TxnContext() *MockImmutableMessage_TxnContext_Call {
	return &MockImmutableMessage_TxnContext_Call{Call: _e.mock.On("TxnContext")}
}

func (_c *MockImmutableMessage_TxnContext_Call) Run(run func()) *MockImmutableMessage_TxnContext_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockImmutableMessage_TxnContext_Call) Return(_a0 *message.TxnContext) *MockImmutableMessage_TxnContext_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockImmutableMessage_TxnContext_Call) RunAndReturn(run func() *message.TxnContext) *MockImmutableMessage_TxnContext_Call {
	_c.Call.Return(run)
	return _c
}

// VChannel provides a mock function with given fields:
func (_m *MockImmutableMessage) VChannel() string {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for VChannel")
	}

	var r0 string
	if rf, ok := ret.Get(0).(func() string); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(string)
	}

	return r0
}

// MockImmutableMessage_VChannel_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'VChannel'
type MockImmutableMessage_VChannel_Call struct {
	*mock.Call
}

// VChannel is a helper method to define mock.On call
func (_e *MockImmutableMessage_Expecter) VChannel() *MockImmutableMessage_VChannel_Call {
	return &MockImmutableMessage_VChannel_Call{Call: _e.mock.On("VChannel")}
}

func (_c *MockImmutableMessage_VChannel_Call) Run(run func()) *MockImmutableMessage_VChannel_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockImmutableMessage_VChannel_Call) Return(_a0 string) *MockImmutableMessage_VChannel_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockImmutableMessage_VChannel_Call) RunAndReturn(run func() string) *MockImmutableMessage_VChannel_Call {
	_c.Call.Return(run)
	return _c
}

// Version provides a mock function with given fields:
func (_m *MockImmutableMessage) Version() message.Version {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for Version")
	}

	var r0 message.Version
	if rf, ok := ret.Get(0).(func() message.Version); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(message.Version)
	}

	return r0
}

// MockImmutableMessage_Version_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Version'
type MockImmutableMessage_Version_Call struct {
	*mock.Call
}

// Version is a helper method to define mock.On call
func (_e *MockImmutableMessage_Expecter) Version() *MockImmutableMessage_Version_Call {
	return &MockImmutableMessage_Version_Call{Call: _e.mock.On("Version")}
}

func (_c *MockImmutableMessage_Version_Call) Run(run func()) *MockImmutableMessage_Version_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockImmutableMessage_Version_Call) Return(_a0 message.Version) *MockImmutableMessage_Version_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockImmutableMessage_Version_Call) RunAndReturn(run func() message.Version) *MockImmutableMessage_Version_Call {
	_c.Call.Return(run)
	return _c
}

// WALName provides a mock function with given fields:
func (_m *MockImmutableMessage) WALName() string {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for WALName")
	}

	var r0 string
	if rf, ok := ret.Get(0).(func() string); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(string)
	}

	return r0
}

// MockImmutableMessage_WALName_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'WALName'
type MockImmutableMessage_WALName_Call struct {
	*mock.Call
}

// WALName is a helper method to define mock.On call
func (_e *MockImmutableMessage_Expecter) WALName() *MockImmutableMessage_WALName_Call {
	return &MockImmutableMessage_WALName_Call{Call: _e.mock.On("WALName")}
}

func (_c *MockImmutableMessage_WALName_Call) Run(run func()) *MockImmutableMessage_WALName_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockImmutableMessage_WALName_Call) Return(_a0 string) *MockImmutableMessage_WALName_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockImmutableMessage_WALName_Call) RunAndReturn(run func() string) *MockImmutableMessage_WALName_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockImmutableMessage creates a new instance of MockImmutableMessage. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockImmutableMessage(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockImmutableMessage {
	mock := &MockImmutableMessage{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
