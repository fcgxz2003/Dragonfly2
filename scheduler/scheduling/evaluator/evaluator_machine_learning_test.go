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
				assert.Equal(reflect.TypeOf(e).Elem().Name(), "evaluatorML")
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

func TestEvaluatorMachineLearning_float64ToByte(t *testing.T) {
	tests := []struct {
		name   string
		data   float64
		expect func(t *testing.T, o []byte)
	}{
		{
			name: "convert byte to float64",
			data: 0.1,
			expect: func(t *testing.T, o []byte) {
				assert := assert.New(t)
				assert.Equal(o, []byte{0x9a, 0x99, 0x99, 0x99, 0x99, 0x99, 0xb9, 0x3f})
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctl := gomock.NewController(t)
			defer ctl.Finish()
			tc.expect(t, float64ToByte(tc.data))
		})
	}
}

func TestEvaluatorMachineLearning_byteToFloat64(t *testing.T) {
	tests := []struct {
		name   string
		data   []byte
		expect func(t *testing.T, o float64)
	}{
		{
			name: "convert byte to float64",
			data: float64ToByte(0.1),
			expect: func(t *testing.T, o float64) {
				assert := assert.New(t)
				assert.Equal(o, 0.1)
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctl := gomock.NewController(t)
			defer ctl.Finish()
			tc.expect(t, byteToFloat64(tc.data))
		})
	}
}

func TestEvaluatorMachineLearning_parseIP(t *testing.T) {
	tests := []struct {
		name   string
		ip     string
		expect func(t *testing.T, feature []float64)
	}{
		{
			name: "parse ip to feature",
			ip:   "172.0.0.1",
			expect: func(t *testing.T, feature []float64) {
				assert := assert.New(t)
				assert.EqualValues(feature, []float64{
					0, 0, 1, 1, 0, 1, 0, 1,
					0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0,
					1, 0, 0, 0, 0, 0, 0, 0})
			},
		},
		{
			name: "parse ip to feature error",
			ip:   "foo",
			expect: func(t *testing.T, feature []float64) {
				assert := assert.New(t)
				assert.EqualValues(feature, []float64{
					0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0})
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctl := gomock.NewController(t)
			defer ctl.Finish()
			tc.expect(t, parseIP(tc.ip))
		})
	}
}
