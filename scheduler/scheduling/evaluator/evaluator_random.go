/*
 *     Copyright 2020 The Dragonfly Authors
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
	"math/rand"
	"time"

	"d7y.io/dragonfly/v2/scheduler/resource"
)

type evaluatorRandom struct {
	evaluator
}

func newEvaluatorRandom() Evaluator {
	return &evaluatorRandom{}
}

// EvaluateParents sort parents by evaluating multiple feature scores.
func (e *evaluatorRandom) EvaluateParents(parents []*resource.Peer, child *resource.Peer, totalPieceCount int32) []*resource.Peer {
	rand.New(rand.NewSource(time.Now().UnixNano()))
	rand.Shuffle(len(parents), func(i, j int) {
		parents[i], parents[j] = parents[j], parents[i]
	})

	return parents
}
