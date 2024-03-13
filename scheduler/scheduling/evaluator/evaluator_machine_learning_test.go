/*
 *     Copyright 2024 The Dragonfly Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package evaluator

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	gomock "go.uber.org/mock/gomock"

	"d7y.io/dragonfly/v2/pkg/idgen"
	inferenceclientmock "d7y.io/dragonfly/v2/pkg/rpc/inference/client/mocks"
	"d7y.io/dragonfly/v2/pkg/types"
	networktopologymocks "d7y.io/dragonfly/v2/scheduler/networktopology/mocks"
	"d7y.io/dragonfly/v2/scheduler/resource"
)

func TestEvaluatorMachineLearning_newEvaluatorMachineLearning(t *testing.T) {
	tests := []struct {
		name   string
		expect func(t *testing.T, e any)
	}{
		{
			name: "new evaluator commonv1",
			expect: func(t *testing.T, e any) {
				assert := assert.New(t)
				assert.Equal(reflect.TypeOf(e).Elem().Name(), "evaluatorMachineLearning")
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctl := gomock.NewController(t)
			defer ctl.Finish()
			tc.expect(t, newEvaluatorMachineLearning())
		})
	}
}

func TestEvaluatorMachineLearning_aggregationHosts(t *testing.T) {
	mockFirstOrderHosts := []*resource.Host{
		resource.NewHost(idgen.HostIDV2("127.0.0.1", "foo"), "127.0.0.1", "foo", 8003, 8001, types.HostTypeNormal),
		resource.NewHost(idgen.HostIDV2("127.0.0.1", "bar"), "127.0.0.1", "bar", 8003, 8001, types.HostTypeNormal),
	}

	mockSecondOrderHosts := []*resource.Host{
		resource.NewHost(idgen.HostIDV2("127.0.0.1", "baz"), "127.0.0.1", "baz", 8003, 8001, types.HostTypeNormal),
		resource.NewHost(idgen.HostIDV2("127.0.0.1", "bac"), "127.0.0.1", "bac", 8003, 8001, types.HostTypeNormal),
		resource.NewHost(idgen.HostIDV2("127.0.0.1", "bad"), "127.0.0.1", "bad", 8003, 8001, types.HostTypeNormal),
		resource.NewHost(idgen.HostIDV2("127.0.0.1", "bae"), "127.0.0.1", "bae", 8003, 8001, types.HostTypeNormal),
	}

	tests := []struct {
		name   string
		mock   func(nt *networktopologymocks.MockNetworkTopologyMockRecorder)
		expect func(t *testing.T, e Evaluator)
	}{
		{
			name: "get aggregation hosts success",
			mock: func(nt *networktopologymocks.MockNetworkTopologyMockRecorder) {
				gomock.InOrder(
					nt.Neighbours(gomock.Eq(&mockRawHost), 2).Return(mockFirstOrderHosts, nil).Times(1),
					nt.Neighbours(gomock.Eq(mockFirstOrderHosts[0]), 2).Return([]*resource.Host{mockSecondOrderHosts[0], mockSecondOrderHosts[1]}, nil).Times(1),
					nt.Neighbours(gomock.Eq(mockFirstOrderHosts[1]), 2).Return([]*resource.Host{mockSecondOrderHosts[2], mockSecondOrderHosts[3]}, nil).Times(1),
				)
			},
			expect: func(t *testing.T, e Evaluator) {
				assert := assert.New(t)
				firstOrderNeighbours, secondOrderNeighbours, err := e.(*evaluatorMachineLearning).aggregationHosts(&mockRawHost, 2)
				assert.Nil(err)
				assert.EqualValues(len(firstOrderNeighbours), 2)
				assert.EqualValues(len(secondOrderNeighbours), 2)
				assert.EqualValues(len(secondOrderNeighbours[0]), 2)
				assert.EqualValues(len(secondOrderNeighbours[1]), 2)
				assert.EqualValues(firstOrderNeighbours[0], mockFirstOrderHosts[0])
				assert.EqualValues(firstOrderNeighbours[1], mockFirstOrderHosts[1])
				assert.EqualValues(secondOrderNeighbours[0][0], mockSecondOrderHosts[0])
				assert.EqualValues(secondOrderNeighbours[0][1], mockSecondOrderHosts[1])
				assert.EqualValues(secondOrderNeighbours[1][0], mockSecondOrderHosts[2])
				assert.EqualValues(secondOrderNeighbours[1][1], mockSecondOrderHosts[3])
			},
		},
		{
			name: "get no enough first aggregation hosts",
			mock: func(nt *networktopologymocks.MockNetworkTopologyMockRecorder) {
				gomock.InOrder(
					nt.Neighbours(gomock.Eq(&mockRawHost), 2).Return([]*resource.Host{mockFirstOrderHosts[0]}, nil).Times(1),
					nt.Neighbours(gomock.Eq(mockFirstOrderHosts[0]), 2).Return([]*resource.Host{mockSecondOrderHosts[0], mockSecondOrderHosts[1]}, nil).Times(2),
				)
			},
			expect: func(t *testing.T, e Evaluator) {
				assert := assert.New(t)
				firstOrderNeighbours, secondOrderNeighbours, err := e.(*evaluatorMachineLearning).aggregationHosts(&mockRawHost, 2)
				assert.Nil(err)
				assert.EqualValues(len(firstOrderNeighbours), 2)
				assert.EqualValues(len(secondOrderNeighbours), 2)
				assert.EqualValues(len(secondOrderNeighbours[0]), 2)
				assert.EqualValues(len(secondOrderNeighbours[1]), 2)
				assert.EqualValues(firstOrderNeighbours[0], mockFirstOrderHosts[0])
				assert.EqualValues(firstOrderNeighbours[1], mockFirstOrderHosts[0])
				assert.EqualValues(secondOrderNeighbours[0][0], mockSecondOrderHosts[0])
				assert.EqualValues(secondOrderNeighbours[0][1], mockSecondOrderHosts[1])
				assert.EqualValues(secondOrderNeighbours[1][0], mockSecondOrderHosts[0])
				assert.EqualValues(secondOrderNeighbours[1][1], mockSecondOrderHosts[1])
			},
		},
		{
			name: "get no enough second aggregation hosts",
			mock: func(nt *networktopologymocks.MockNetworkTopologyMockRecorder) {
				gomock.InOrder(
					nt.Neighbours(gomock.Eq(&mockRawHost), 2).Return(mockFirstOrderHosts, nil).Times(1),
					nt.Neighbours(gomock.Eq(mockFirstOrderHosts[0]), 2).Return([]*resource.Host{mockSecondOrderHosts[0]}, nil).Times(1),
					nt.Neighbours(gomock.Eq(mockFirstOrderHosts[1]), 2).Return([]*resource.Host{mockSecondOrderHosts[2]}, nil).Times(1),
				)
			},
			expect: func(t *testing.T, e Evaluator) {
				assert := assert.New(t)
				firstOrderNeighbours, secondOrderNeighbours, err := e.(*evaluatorMachineLearning).aggregationHosts(&mockRawHost, 2)
				assert.Nil(err)
				assert.EqualValues(len(firstOrderNeighbours), 2)
				assert.EqualValues(len(secondOrderNeighbours), 2)
				assert.EqualValues(len(secondOrderNeighbours[0]), 2)
				assert.EqualValues(len(secondOrderNeighbours[1]), 2)
				assert.EqualValues(firstOrderNeighbours[0], mockFirstOrderHosts[0])
				assert.EqualValues(firstOrderNeighbours[1], mockFirstOrderHosts[1])
				assert.EqualValues(secondOrderNeighbours[0][0], mockSecondOrderHosts[0])
				assert.EqualValues(secondOrderNeighbours[0][1], mockSecondOrderHosts[0])
				assert.EqualValues(secondOrderNeighbours[1][0], mockSecondOrderHosts[2])
				assert.EqualValues(secondOrderNeighbours[1][1], mockSecondOrderHosts[2])
			},
		},
		{
			name: "get empty aggregation hosts",
			mock: func(nt *networktopologymocks.MockNetworkTopologyMockRecorder) {
				gomock.InOrder(
					nt.Neighbours(gomock.Eq(&mockRawHost), 2).Return([]*resource.Host{}, nil).Times(1),
					nt.Neighbours(gomock.Eq(&mockRawHost), 2).Return([]*resource.Host{}, nil).Times(1),
					nt.Neighbours(gomock.Eq(&mockRawHost), 2).Return([]*resource.Host{}, nil).Times(1),
				)
			},
			expect: func(t *testing.T, e Evaluator) {
				assert := assert.New(t)
				firstOrderNeighbours, secondOrderNeighbours, err := e.(*evaluatorMachineLearning).aggregationHosts(&mockRawHost, 2)
				assert.Nil(err)
				assert.EqualValues(len(firstOrderNeighbours), 2)
				assert.EqualValues(len(secondOrderNeighbours), 2)
				assert.EqualValues(len(secondOrderNeighbours[0]), 2)
				assert.EqualValues(len(secondOrderNeighbours[1]), 2)
				assert.EqualValues(firstOrderNeighbours[0], &mockRawHost)
				assert.EqualValues(firstOrderNeighbours[1], &mockRawHost)
				assert.EqualValues(secondOrderNeighbours[0][0], &mockRawHost)
				assert.EqualValues(secondOrderNeighbours[0][1], &mockRawHost)
				assert.EqualValues(secondOrderNeighbours[1][0], &mockRawHost)
				assert.EqualValues(secondOrderNeighbours[1][1], &mockRawHost)
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctl := gomock.NewController(t)
			defer ctl.Finish()
			mockNetworkTopology := networktopologymocks.NewMockNetworkTopology(ctl)
			mockInferenceClient := inferenceclientmock.NewMockV1(ctl)
			tc.mock(mockNetworkTopology.EXPECT())
			e := newEvaluatorMachineLearning(WithNetworkTopologyInMachineLearning(mockNetworkTopology), WithInferenceClient(mockInferenceClient))
			tc.expect(t, e)
		})
	}
}

func TestEvaluatorMachineLearning_parseIP(t *testing.T) {
	tests := []struct {
		name   string
		ip     string
		expect func(t *testing.T, feature []float32, err error)
	}{
		{
			name: "parse ip to feature",
			ip:   "172.0.0.1",
			expect: func(t *testing.T, feature []float32, err error) {
				assert := assert.New(t)
				assert.Nil(err)
				assert.EqualValues(feature, []float32{
					0, 0, 1, 1, 0, 1, 0, 1,
					0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0,
					1, 0, 0, 0, 0, 0, 0, 0})
			},
		},
		{
			name: "parse ip to feature error",
			ip:   "foo",
			expect: func(t *testing.T, feature []float32, err error) {
				assert := assert.New(t)
				assert.Nil(feature)
				assert.EqualError(err, "prase ip error")
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctl := gomock.NewController(t)
			defer ctl.Finish()
			feature, err := parseIP(tc.ip)
			tc.expect(t, feature, err)
		})
	}
}
