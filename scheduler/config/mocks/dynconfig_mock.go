// Code generated by MockGen. DO NOT EDIT.
// Source: dynconfig.go
//
// Generated by this command:
//
//	mockgen -destination mocks/dynconfig_mock.go -source dynconfig.go -package mocks
//

// Package mocks is a generated GoMock package.
package mocks

import (
	reflect "reflect"

	types "d7y.io/dragonfly/v2/manager/types"
	config "d7y.io/dragonfly/v2/scheduler/config"
	manager "github.com/fcgxz2003/api/v2/pkg/apis/manager/v2"
	gomock "go.uber.org/mock/gomock"
	resolver "google.golang.org/grpc/resolver"
)

// MockDynconfigInterface is a mock of DynconfigInterface interface.
type MockDynconfigInterface struct {
	ctrl     *gomock.Controller
	recorder *MockDynconfigInterfaceMockRecorder
}

// MockDynconfigInterfaceMockRecorder is the mock recorder for MockDynconfigInterface.
type MockDynconfigInterfaceMockRecorder struct {
	mock *MockDynconfigInterface
}

// NewMockDynconfigInterface creates a new mock instance.
func NewMockDynconfigInterface(ctrl *gomock.Controller) *MockDynconfigInterface {
	mock := &MockDynconfigInterface{ctrl: ctrl}
	mock.recorder = &MockDynconfigInterfaceMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockDynconfigInterface) EXPECT() *MockDynconfigInterfaceMockRecorder {
	return m.recorder
}

// Deregister mocks base method.
func (m *MockDynconfigInterface) Deregister(arg0 config.Observer) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "Deregister", arg0)
}

// Deregister indicates an expected call of Deregister.
func (mr *MockDynconfigInterfaceMockRecorder) Deregister(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Deregister", reflect.TypeOf((*MockDynconfigInterface)(nil).Deregister), arg0)
}

// Get mocks base method.
func (m *MockDynconfigInterface) Get() (*config.DynconfigData, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Get")
	ret0, _ := ret[0].(*config.DynconfigData)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// Get indicates an expected call of Get.
func (mr *MockDynconfigInterfaceMockRecorder) Get() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Get", reflect.TypeOf((*MockDynconfigInterface)(nil).Get))
}

// GetApplications mocks base method.
func (m *MockDynconfigInterface) GetApplications() ([]*manager.Application, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetApplications")
	ret0, _ := ret[0].([]*manager.Application)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetApplications indicates an expected call of GetApplications.
func (mr *MockDynconfigInterfaceMockRecorder) GetApplications() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetApplications", reflect.TypeOf((*MockDynconfigInterface)(nil).GetApplications))
}

// GetResolveSeedPeerAddrs mocks base method.
func (m *MockDynconfigInterface) GetResolveSeedPeerAddrs() ([]resolver.Address, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetResolveSeedPeerAddrs")
	ret0, _ := ret[0].([]resolver.Address)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetResolveSeedPeerAddrs indicates an expected call of GetResolveSeedPeerAddrs.
func (mr *MockDynconfigInterfaceMockRecorder) GetResolveSeedPeerAddrs() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetResolveSeedPeerAddrs", reflect.TypeOf((*MockDynconfigInterface)(nil).GetResolveSeedPeerAddrs))
}

// GetScheduler mocks base method.
func (m *MockDynconfigInterface) GetScheduler() (*manager.Scheduler, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetScheduler")
	ret0, _ := ret[0].(*manager.Scheduler)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetScheduler indicates an expected call of GetScheduler.
func (mr *MockDynconfigInterfaceMockRecorder) GetScheduler() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetScheduler", reflect.TypeOf((*MockDynconfigInterface)(nil).GetScheduler))
}

// GetSchedulerCluster mocks base method.
func (m *MockDynconfigInterface) GetSchedulerCluster() (*manager.SchedulerCluster, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetSchedulerCluster")
	ret0, _ := ret[0].(*manager.SchedulerCluster)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetSchedulerCluster indicates an expected call of GetSchedulerCluster.
func (mr *MockDynconfigInterfaceMockRecorder) GetSchedulerCluster() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetSchedulerCluster", reflect.TypeOf((*MockDynconfigInterface)(nil).GetSchedulerCluster))
}

// GetSchedulerClusterClientConfig mocks base method.
func (m *MockDynconfigInterface) GetSchedulerClusterClientConfig() (types.SchedulerClusterClientConfig, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetSchedulerClusterClientConfig")
	ret0, _ := ret[0].(types.SchedulerClusterClientConfig)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetSchedulerClusterClientConfig indicates an expected call of GetSchedulerClusterClientConfig.
func (mr *MockDynconfigInterfaceMockRecorder) GetSchedulerClusterClientConfig() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetSchedulerClusterClientConfig", reflect.TypeOf((*MockDynconfigInterface)(nil).GetSchedulerClusterClientConfig))
}

// GetSchedulerClusterConfig mocks base method.
func (m *MockDynconfigInterface) GetSchedulerClusterConfig() (types.SchedulerClusterConfig, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetSchedulerClusterConfig")
	ret0, _ := ret[0].(types.SchedulerClusterConfig)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetSchedulerClusterConfig indicates an expected call of GetSchedulerClusterConfig.
func (mr *MockDynconfigInterfaceMockRecorder) GetSchedulerClusterConfig() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetSchedulerClusterConfig", reflect.TypeOf((*MockDynconfigInterface)(nil).GetSchedulerClusterConfig))
}

// GetSeedPeers mocks base method.
func (m *MockDynconfigInterface) GetSeedPeers() ([]*manager.SeedPeer, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetSeedPeers")
	ret0, _ := ret[0].([]*manager.SeedPeer)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetSeedPeers indicates an expected call of GetSeedPeers.
func (mr *MockDynconfigInterfaceMockRecorder) GetSeedPeers() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetSeedPeers", reflect.TypeOf((*MockDynconfigInterface)(nil).GetSeedPeers))
}

// Notify mocks base method.
func (m *MockDynconfigInterface) Notify() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Notify")
	ret0, _ := ret[0].(error)
	return ret0
}

// Notify indicates an expected call of Notify.
func (mr *MockDynconfigInterfaceMockRecorder) Notify() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Notify", reflect.TypeOf((*MockDynconfigInterface)(nil).Notify))
}

// Refresh mocks base method.
func (m *MockDynconfigInterface) Refresh() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Refresh")
	ret0, _ := ret[0].(error)
	return ret0
}

// Refresh indicates an expected call of Refresh.
func (mr *MockDynconfigInterfaceMockRecorder) Refresh() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Refresh", reflect.TypeOf((*MockDynconfigInterface)(nil).Refresh))
}

// Register mocks base method.
func (m *MockDynconfigInterface) Register(arg0 config.Observer) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "Register", arg0)
}

// Register indicates an expected call of Register.
func (mr *MockDynconfigInterfaceMockRecorder) Register(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Register", reflect.TypeOf((*MockDynconfigInterface)(nil).Register), arg0)
}

// Serve mocks base method.
func (m *MockDynconfigInterface) Serve() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Serve")
	ret0, _ := ret[0].(error)
	return ret0
}

// Serve indicates an expected call of Serve.
func (mr *MockDynconfigInterfaceMockRecorder) Serve() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Serve", reflect.TypeOf((*MockDynconfigInterface)(nil).Serve))
}

// Stop mocks base method.
func (m *MockDynconfigInterface) Stop() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Stop")
	ret0, _ := ret[0].(error)
	return ret0
}

// Stop indicates an expected call of Stop.
func (mr *MockDynconfigInterfaceMockRecorder) Stop() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Stop", reflect.TypeOf((*MockDynconfigInterface)(nil).Stop))
}

// MockObserver is a mock of Observer interface.
type MockObserver struct {
	ctrl     *gomock.Controller
	recorder *MockObserverMockRecorder
}

// MockObserverMockRecorder is the mock recorder for MockObserver.
type MockObserverMockRecorder struct {
	mock *MockObserver
}

// NewMockObserver creates a new mock instance.
func NewMockObserver(ctrl *gomock.Controller) *MockObserver {
	mock := &MockObserver{ctrl: ctrl}
	mock.recorder = &MockObserverMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockObserver) EXPECT() *MockObserverMockRecorder {
	return m.recorder
}

// OnNotify mocks base method.
func (m *MockObserver) OnNotify(arg0 *config.DynconfigData) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "OnNotify", arg0)
}

// OnNotify indicates an expected call of OnNotify.
func (mr *MockObserverMockRecorder) OnNotify(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "OnNotify", reflect.TypeOf((*MockObserver)(nil).OnNotify), arg0)
}
