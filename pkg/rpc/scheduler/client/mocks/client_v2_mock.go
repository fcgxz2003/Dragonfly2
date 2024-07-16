// Code generated by MockGen. DO NOT EDIT.
// Source: client_v2.go
//
// Generated by this command:
//
//	mockgen -destination mocks/client_v2_mock.go -source client_v2.go -package mocks
//

// Package mocks is a generated GoMock package.
package mocks

import (
	context "context"
	reflect "reflect"

	common "github.com/fcgxz2003/api/v2/pkg/apis/common/v2"
	scheduler "github.com/fcgxz2003/api/v2/pkg/apis/scheduler/v2"
	gomock "go.uber.org/mock/gomock"
	grpc "google.golang.org/grpc"
)

// MockV2 is a mock of V2 interface.
type MockV2 struct {
	ctrl     *gomock.Controller
	recorder *MockV2MockRecorder
}

// MockV2MockRecorder is the mock recorder for MockV2.
type MockV2MockRecorder struct {
	mock *MockV2
}

// NewMockV2 creates a new mock instance.
func NewMockV2(ctrl *gomock.Controller) *MockV2 {
	mock := &MockV2{ctrl: ctrl}
	mock.recorder = &MockV2MockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockV2) EXPECT() *MockV2MockRecorder {
	return m.recorder
}

// AnnounceHost mocks base method.
func (m *MockV2) AnnounceHost(arg0 context.Context, arg1 *scheduler.AnnounceHostRequest, arg2 ...grpc.CallOption) error {
	m.ctrl.T.Helper()
	varargs := []any{arg0, arg1}
	for _, a := range arg2 {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "AnnounceHost", varargs...)
	ret0, _ := ret[0].(error)
	return ret0
}

// AnnounceHost indicates an expected call of AnnounceHost.
func (mr *MockV2MockRecorder) AnnounceHost(arg0, arg1 any, arg2 ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{arg0, arg1}, arg2...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "AnnounceHost", reflect.TypeOf((*MockV2)(nil).AnnounceHost), varargs...)
}

// AnnouncePeer mocks base method.
func (m *MockV2) AnnouncePeer(arg0 context.Context, arg1 string, arg2 ...grpc.CallOption) (scheduler.Scheduler_AnnouncePeerClient, error) {
	m.ctrl.T.Helper()
	varargs := []any{arg0, arg1}
	for _, a := range arg2 {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "AnnouncePeer", varargs...)
	ret0, _ := ret[0].(scheduler.Scheduler_AnnouncePeerClient)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// AnnouncePeer indicates an expected call of AnnouncePeer.
func (mr *MockV2MockRecorder) AnnouncePeer(arg0, arg1 any, arg2 ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{arg0, arg1}, arg2...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "AnnouncePeer", reflect.TypeOf((*MockV2)(nil).AnnouncePeer), varargs...)
}

// Close mocks base method.
func (m *MockV2) Close() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Close")
	ret0, _ := ret[0].(error)
	return ret0
}

// Close indicates an expected call of Close.
func (mr *MockV2MockRecorder) Close() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Close", reflect.TypeOf((*MockV2)(nil).Close))
}

// DeleteHost mocks base method.
func (m *MockV2) DeleteHost(arg0 context.Context, arg1 *scheduler.DeleteHostRequest, arg2 ...grpc.CallOption) error {
	m.ctrl.T.Helper()
	varargs := []any{arg0, arg1}
	for _, a := range arg2 {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "DeleteHost", varargs...)
	ret0, _ := ret[0].(error)
	return ret0
}

// DeleteHost indicates an expected call of DeleteHost.
func (mr *MockV2MockRecorder) DeleteHost(arg0, arg1 any, arg2 ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{arg0, arg1}, arg2...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeleteHost", reflect.TypeOf((*MockV2)(nil).DeleteHost), varargs...)
}

// DeletePeer mocks base method.
func (m *MockV2) DeletePeer(arg0 context.Context, arg1 *scheduler.DeletePeerRequest, arg2 ...grpc.CallOption) error {
	m.ctrl.T.Helper()
	varargs := []any{arg0, arg1}
	for _, a := range arg2 {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "DeletePeer", varargs...)
	ret0, _ := ret[0].(error)
	return ret0
}

// DeletePeer indicates an expected call of DeletePeer.
func (mr *MockV2MockRecorder) DeletePeer(arg0, arg1 any, arg2 ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{arg0, arg1}, arg2...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeletePeer", reflect.TypeOf((*MockV2)(nil).DeletePeer), varargs...)
}

// DeleteTask mocks base method.
func (m *MockV2) DeleteTask(arg0 context.Context, arg1 *scheduler.DeleteTaskRequest, arg2 ...grpc.CallOption) error {
	m.ctrl.T.Helper()
	varargs := []any{arg0, arg1}
	for _, a := range arg2 {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "DeleteTask", varargs...)
	ret0, _ := ret[0].(error)
	return ret0
}

// DeleteTask indicates an expected call of DeleteTask.
func (mr *MockV2MockRecorder) DeleteTask(arg0, arg1 any, arg2 ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{arg0, arg1}, arg2...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeleteTask", reflect.TypeOf((*MockV2)(nil).DeleteTask), varargs...)
}

// StatPeer mocks base method.
func (m *MockV2) StatPeer(arg0 context.Context, arg1 *scheduler.StatPeerRequest, arg2 ...grpc.CallOption) (*common.Peer, error) {
	m.ctrl.T.Helper()
	varargs := []any{arg0, arg1}
	for _, a := range arg2 {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "StatPeer", varargs...)
	ret0, _ := ret[0].(*common.Peer)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// StatPeer indicates an expected call of StatPeer.
func (mr *MockV2MockRecorder) StatPeer(arg0, arg1 any, arg2 ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{arg0, arg1}, arg2...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "StatPeer", reflect.TypeOf((*MockV2)(nil).StatPeer), varargs...)
}

// StatTask mocks base method.
func (m *MockV2) StatTask(arg0 context.Context, arg1 *scheduler.StatTaskRequest, arg2 ...grpc.CallOption) (*common.Task, error) {
	m.ctrl.T.Helper()
	varargs := []any{arg0, arg1}
	for _, a := range arg2 {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "StatTask", varargs...)
	ret0, _ := ret[0].(*common.Task)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// StatTask indicates an expected call of StatTask.
func (mr *MockV2MockRecorder) StatTask(arg0, arg1 any, arg2 ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{arg0, arg1}, arg2...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "StatTask", reflect.TypeOf((*MockV2)(nil).StatTask), varargs...)
}

// SyncProbes mocks base method.
func (m *MockV2) SyncProbes(arg0 context.Context, arg1 *scheduler.SyncProbesRequest, arg2 ...grpc.CallOption) (scheduler.Scheduler_SyncProbesClient, error) {
	m.ctrl.T.Helper()
	varargs := []any{arg0, arg1}
	for _, a := range arg2 {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "SyncProbes", varargs...)
	ret0, _ := ret[0].(scheduler.Scheduler_SyncProbesClient)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// SyncProbes indicates an expected call of SyncProbes.
func (mr *MockV2MockRecorder) SyncProbes(arg0, arg1 any, arg2 ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{arg0, arg1}, arg2...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "SyncProbes", reflect.TypeOf((*MockV2)(nil).SyncProbes), varargs...)
}
