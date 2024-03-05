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
	"context"
	"encoding/binary"
	"math"
	"math/big"
	"net"
	"sort"
	"strings"
	"unsafe"

	"github.com/montanaflynn/stats"

	triton "d7y.io/api/v2/pkg/apis/inference"
	logger "d7y.io/dragonfly/v2/internal/dflog"
	mathematics "d7y.io/dragonfly/v2/pkg/math"
	inferenceclient "d7y.io/dragonfly/v2/pkg/rpc/inference/client"
	"d7y.io/dragonfly/v2/pkg/types"
	"d7y.io/dragonfly/v2/scheduler/networktopology"
	"d7y.io/dragonfly/v2/scheduler/resource"
)

const (
	// Default number of aggregated neighbours.
	defaultAggregationNumber = 3

	// Number of bytes occupied by float64.
	sizeFloat64 = unsafe.Sizeof(float64(0))
)

// evaluatorMachineLearning is an implementation of Evaluator.
type evaluatorMachineLearning struct {
	evaluator
	inferenceClient inferenceclient.V1
	networkTopology networktopology.NetworkTopology
}

// MachineLearningOption is a functional option for configuring the evaluatorMachineLearning.
type MachineLearningOption func(e *evaluatorMachineLearning)

// WithInferenceClient sets the inference client.
func WithInferenceClient(client inferenceclient.V1) MachineLearningOption {
	return func(e *evaluatorMachineLearning) {
		e.inferenceClient = client
	}
}

// WithNetworkTopology sets the network topology.
func WithNetworkTopologyInMachineLearning(networkTopology networktopology.NetworkTopology) MachineLearningOption {
	return func(e *evaluatorMachineLearning) {
		e.networkTopology = networkTopology
	}
}

func newEvaluatorMachineLearning(options ...MachineLearningOption) Evaluator {
	e := &evaluatorMachineLearning{}
	for _, opt := range options {
		opt(e)
	}

	return e
}

// EvaluateParents sort parents by evaluating multiple feature scores.
func (e *evaluatorMachineLearning) EvaluateParents(parents []*resource.Peer, child *resource.Peer, totalPieceCount int32) []*resource.Peer {
	scores := e.evaluate(parents, child, totalPieceCount)
	sort.Slice(
		parents,
		func(i, j int) bool {
			return scores[i] > scores[j]
		},
	)

	return parents
}

// The larger the value, the higher the priority.
func (e *evaluatorMachineLearning) evaluate(parents []*resource.Peer, child *resource.Peer, totalPieceCount int32) []float64 {
	scores, err := e.inference(parents, child)
	if err != nil {
		logger.Info("using evaluate base algorithm")
		scores := make([]float64, len(parents))
		for i, parent := range parents {
			scores[i] = e.evaluateBase(parent, child, totalPieceCount)
		}

		return scores
	}

	logger.Info("using machine learining algorithm")
	return scores
}

// The larger the value, the higher the priority.
func (e *evaluatorMachineLearning) evaluateBase(parent *resource.Peer, child *resource.Peer, totalPieceCount int32) float64 {
	parentLocation := parent.Host.Network.Location
	parentIDC := parent.Host.Network.IDC
	childLocation := child.Host.Network.Location
	childIDC := child.Host.Network.IDC

	return finishedPieceWeight*e.calculatePieceScore(parent, child, totalPieceCount) +
		parentHostUploadSuccessWeight*e.calculateParentHostUploadSuccessScore(parent) +
		freeUploadWeight*e.calculateFreeUploadScore(parent.Host) +
		hostTypeWeight*e.calculateHostTypeScore(parent) +
		idcAffinityWeight*e.calculateIDCAffinityScore(parentIDC, childIDC) +
		locationAffinityWeight*e.calculateMultiElementAffinityScore(parentLocation, childLocation)
}

// calculatePieceScore 0.0~unlimited larger and better.
func (e *evaluatorMachineLearning) calculatePieceScore(parent *resource.Peer, child *resource.Peer, totalPieceCount int32) float64 {
	// If the total piece is determined, normalize the number of
	// pieces downloaded by the parent node.
	if totalPieceCount > 0 {
		finishedPieceCount := parent.FinishedPieces.Count()
		return float64(finishedPieceCount) / float64(totalPieceCount)
	}

	// Use the difference between the parent node and the child node to
	// download the piece to roughly represent the piece score.
	parentFinishedPieceCount := parent.FinishedPieces.Count()
	childFinishedPieceCount := child.FinishedPieces.Count()
	return float64(parentFinishedPieceCount) - float64(childFinishedPieceCount)
}

// calculateParentHostUploadSuccessScore 0.0~unlimited larger and better.
func (e *evaluatorMachineLearning) calculateParentHostUploadSuccessScore(peer *resource.Peer) float64 {
	uploadCount := peer.Host.UploadCount.Load()
	uploadFailedCount := peer.Host.UploadFailedCount.Load()
	if uploadCount < uploadFailedCount {
		return minScore
	}

	// Host has not been scheduled, then it is scheduled first.
	if uploadCount == 0 && uploadFailedCount == 0 {
		return maxScore
	}

	return float64(uploadCount-uploadFailedCount) / float64(uploadCount)
}

// calculateFreeUploadScore 0.0~1.0 larger and better.
func (e *evaluatorMachineLearning) calculateFreeUploadScore(host *resource.Host) float64 {
	ConcurrentUploadLimit := host.ConcurrentUploadLimit.Load()
	freeUploadCount := host.FreeUploadCount()
	if ConcurrentUploadLimit > 0 && freeUploadCount > 0 {
		return float64(freeUploadCount) / float64(ConcurrentUploadLimit)
	}

	return minScore
}

// calculateHostTypeScore 0.0~1.0 larger and better.
func (e *evaluatorMachineLearning) calculateHostTypeScore(peer *resource.Peer) float64 {
	// When the task is downloaded for the first time,
	// peer will be scheduled to seed peer first,
	// otherwise it will be scheduled to dfdaemon first.
	if peer.Host.Type != types.HostTypeNormal {
		if peer.FSM.Is(resource.PeerStateReceivedNormal) ||
			peer.FSM.Is(resource.PeerStateRunning) {
			return maxScore
		}

		return minScore
	}

	return maxScore * 0.5
}

// calculateIDCAffinityScore 0.0~1.0 larger and better.
func (e *evaluatorMachineLearning) calculateIDCAffinityScore(dst, src string) float64 {
	if dst == "" || src == "" {
		return minScore
	}

	if strings.EqualFold(dst, src) {
		return maxScore
	}

	return minScore
}

// calculateMultiElementAffinityScore 0.0~1.0 larger and better.
func (e *evaluatorMachineLearning) calculateMultiElementAffinityScore(dst, src string) float64 {
	if dst == "" || src == "" {
		return minScore
	}

	if strings.EqualFold(dst, src) {
		return maxScore
	}

	// Calculate the number of multi-element matches divided by "|".
	var score, elementLen int
	dstElements := strings.Split(dst, types.AffinitySeparator)
	srcElements := strings.Split(src, types.AffinitySeparator)
	elementLen = mathematics.Min(len(dstElements), len(srcElements))

	// Maximum element length is 5.
	if elementLen > maxElementLen {
		elementLen = maxElementLen
	}

	for i := 0; i < elementLen; i++ {
		if !strings.EqualFold(dstElements[i], srcElements[i]) {
			break
		}

		score++
	}

	return float64(score) / float64(maxElementLen)
}

func (e *evaluatorMachineLearning) IsBadNode(peer *resource.Peer) bool {
	if peer.FSM.Is(resource.PeerStateFailed) || peer.FSM.Is(resource.PeerStateLeave) || peer.FSM.Is(resource.PeerStatePending) ||
		peer.FSM.Is(resource.PeerStateReceivedTiny) || peer.FSM.Is(resource.PeerStateReceivedSmall) ||
		peer.FSM.Is(resource.PeerStateReceivedNormal) || peer.FSM.Is(resource.PeerStateReceivedEmpty) {
		peer.Log.Debugf("peer is bad node because peer status is %s", peer.FSM.Current())
		return true
	}

	// Determine whether to bad node based on piece download costs.
	costs := stats.LoadRawData(peer.PieceCosts())
	len := len(costs)
	// Peer has not finished downloading enough piece.
	if len < minAvailableCostLen {
		logger.Debugf("peer %s has not finished downloading enough piece, it can't be bad node", peer.ID)
		return false
	}

	lastCost := costs[len-1]
	mean, _ := stats.Mean(costs[:len-1]) // nolint: errcheck

	// Download costs does not meet the normal distribution,
	// if the last cost is twenty times more than mean, it is bad node.
	if len < normalDistributionLen {
		isBadNode := big.NewFloat(lastCost).Cmp(big.NewFloat(mean*20)) > 0
		logger.Debugf("peer %s mean is %.2f and it is bad node: %t", peer.ID, mean, isBadNode)
		return isBadNode
	}

	// Download costs satisfies the normal distribution,
	// last cost falling outside of three-sigma effect need to be adjusted parent,
	// refer to https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule.
	stdev, _ := stats.StandardDeviation(costs[:len-1]) // nolint: errcheck
	isBadNode := big.NewFloat(lastCost).Cmp(big.NewFloat(mean+3*stdev)) > 0
	logger.Debugf("peer %s meet the normal distribution, costs mean is %.2f and standard deviation is %.2f, peer is bad node: %t",
		peer.ID, mean, stdev, isBadNode)
	return isBadNode
}

func (e *evaluatorMachineLearning) inference(parents []*resource.Peer, child *resource.Peer) ([]float64, error) {
	// Find the aggregation hosts for child.
	childNeighbours, err := e.networkTopology.Neighbours(child.Host, defaultAggregationNumber)
	if err != nil {
		return []float64{}, err
	}

	childNeighboursNeighbours := make([][]*resource.Host, 0, defaultAggregationNumber)
	for _, childNeighbour := range childNeighbours {
		neighbourNeighbours, err := e.networkTopology.Neighbours(childNeighbour, defaultAggregationNumber)
		if err != nil {
			return []float64{}, err
		}

		childNeighboursNeighbours = append(childNeighboursNeighbours, neighbourNeighbours)
	}

	// Generate feature vector for child and its aggregation hosts.
	// childFeatures := parseIP(child.Host.IP)

	// Find the aggregation hosts for parents.
	parentsNeighbours := make([][]*resource.Host, 0, defaultAggregationNumber)
	parentsNeighboursNeighbours := make([][][]*resource.Host, 0, defaultAggregationNumber)
	for _, parent := range parents {
		neighbours, err := e.networkTopology.Neighbours(parent.Host, defaultAggregationNumber)
		if err != nil {
			return []float64{}, err
		}
		parentsNeighbours = append(parentsNeighbours, neighbours)

		parentNeighbourNeighbours := make([][]*resource.Host, 0, defaultAggregationNumber)
		for _, neighbour := range neighbours {
			neighbourNeighbours, err := e.networkTopology.Neighbours(neighbour, defaultAggregationNumber)
			if err != nil {
				return []float64{}, err
			}

			parentNeighbourNeighbours = append(parentNeighbourNeighbours, neighbourNeighbours)
		}
		parentsNeighboursNeighbours = append(parentsNeighboursNeighbours, parentNeighbourNeighbours)
	}

	// Generate feature vector for parents and theirs aggregation hosts.
	fakeInput := make([][]float64, 0)
	for i := 0; i < len(parents); i++ {
		fakeInput = append(fakeInput, []float64{0.5, 0.5, 0.5, 0.5, 0.8})
	}

	inferInputs := []*triton.ModelInferRequest_InferInputTensor{
		{
			Name:     "inputs",
			Datatype: "FP64",
			Shape:    []int64{int64(len(parents)), 5},
		},
	}

	inferOutputs := []*triton.ModelInferRequest_InferRequestedOutputTensor{
		{
			Name: "output_0",
		},
	}

	inferRequest := triton.ModelInferRequest{
		ModelName:        "simple",
		ModelVersion:     "1",
		Inputs:           inferInputs,
		Outputs:          inferOutputs,
		RawInputContents: preprocess(fakeInput),
	}

	inferResponse, err := e.inferenceClient.ModelInfer(context.Background(), &inferRequest)
	if err != nil {
		logger.Info(err)
		return []float64{}, err
	}

	outputs := postprocess(inferResponse.RawOutputContents)
	logger.Infof("%#v", outputs)
	return outputs, nil
}

// Convert float64 input data into raw bytes (Little Endian).
func preprocess(inputs [][]float64) [][]byte {
	raw := make([][]byte, len(inputs))
	for i := range raw {
		raw[i] = make([]byte, len(inputs[0])*int(sizeFloat64))
		for j := range inputs[i] {
			offset := j * int(sizeFloat64)
			s := float64ToByte(inputs[i][j])
			copy(raw[i][offset:], s)
		}
	}
	return raw
}

// Convert output's raw bytes into float64 data (Little Endian).
func postprocess(raw [][]byte) []float64 {
	outputs := make([]float64, len(raw[0])/int(sizeFloat64))
	for i := range outputs {
		offset := i * int(sizeFloat64)
		s := raw[0][offset : offset+int(unsafe.Sizeof(float64(0)))]
		outputs[i] = byteToFloat64(s)
	}

	return outputs
}

// Convert float64 to byte.
func float64ToByte(v float64) []byte {
	bits := math.Float64bits(v)
	bts := make([]byte, unsafe.Sizeof(v))
	binary.LittleEndian.PutUint64(bts, bits)
	return bts
}

// Convert byte to float64.
func byteToFloat64(v []byte) float64 {
	bits := binary.LittleEndian.Uint64(v)
	return math.Float64frombits(bits)
}

// parseIP parses an ip address to a feature vector.
func parseIP(ip string) []float64 {
	var features = make([]float64, 32)
	prase := net.ParseIP(ip).To4()
	if prase != nil {
		for i := 0; i < net.IPv4len; i++ {
			d := prase[i]
			for j := 0; j < 8; j++ {
				features[i*8+j] = float64(d & 0x1)
				d = d >> 1
			}

		}
	}

	return features
}
