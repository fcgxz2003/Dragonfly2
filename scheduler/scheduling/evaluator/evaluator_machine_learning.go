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
	"errors"
	"math"
	"math/big"
	"math/rand"
	"net"
	"sort"
	"unsafe"

	"github.com/montanaflynn/stats"

	triton "d7y.io/api/v2/pkg/apis/inference"
	logger "d7y.io/dragonfly/v2/internal/dflog"
	inferenceclient "d7y.io/dragonfly/v2/pkg/rpc/inference/client"
	"d7y.io/dragonfly/v2/scheduler/networktopology"
	"d7y.io/dragonfly/v2/scheduler/resource"
	"d7y.io/dragonfly/v2/scheduler/storage"
)

const (
	// Number of bytes occupied by float32.
	sizeFloat32 = int(unsafe.Sizeof(float32(0)))

	// Default number of aggregated neighbours.
	defaultAggregationNumber = 3

	// IPv4 feature length.
	defaultIPv4FeatureLength = 32

	// Default neighbour ip feature length.
	defaultNeighbourIpFeatureLength = defaultAggregationNumber * defaultIPv4FeatureLength

	// Default neighbour's neighbour ip feature length.
	defaultNeighbourNeighbourIpFeatureLength = defaultAggregationNumber * defaultAggregationNumber * defaultIPv4FeatureLength
)

// evaluatorMachineLearning is an implementation of Evaluator.
type evaluatorMachineLearning struct {
	evaluator
	inferenceClient inferenceclient.V1
	networkTopology networktopology.NetworkTopology
	storage         storage.Storage
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

// WithStorage sets the storage.
func WithStorage(storage storage.Storage) MachineLearningOption {
	return func(e *evaluatorMachineLearning) {
		e.storage = storage
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
		logger.Errorf("machine learining algorithm error:", err)
		scores := make([]float64, len(parents))
		return scores
	}

	logger.Info("using machine learining algorithm")
	return scores
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
	childFirstOrderNeighbours, childSecondOrderNeighbours, err := e.aggregationHosts(child.Host, defaultAggregationNumber)
	if err != nil {
		logger.Error(err)
		return []float64{}, err
	}

	// Generate feature vector for child and its aggregation hosts.
	childIPFeature, err := parseIP(child.Host.IP)
	if err != nil {
		return []float64{}, err
	}

	childNegIPFeatures := make([]float32, 0, defaultNeighbourIpFeatureLength)
	for _, childFirstOrderNeighbour := range childFirstOrderNeighbours {
		childNegIPFeature, err := parseIP(childFirstOrderNeighbour.IP)
		if err != nil {
			return []float64{}, err
		}

		childNegIPFeatures = append(childNegIPFeatures, childNegIPFeature...)
	}

	childNegNegIPFeatures := make([]float32, 0, defaultNeighbourNeighbourIpFeatureLength)
	for _, childSecondOrderNeighbour := range childSecondOrderNeighbours {
		for _, host := range childSecondOrderNeighbour {
			childNegNegIPFeature, err := parseIP(host.IP)
			if err != nil {
				return []float64{}, err
			}

			childNegNegIPFeatures = append(childNegNegIPFeatures, childNegNegIPFeature...)
		}
	}

	var (
		srcFeature       = make([]float32, 0, len(parents)*defaultIPv4FeatureLength)
		srcNegFeature    = make([]float32, 0, len(parents)*defaultNeighbourIpFeatureLength)
		srcNegNegFeature = make([]float32, 0, len(parents)*defaultNeighbourNeighbourIpFeatureLength)
	)

	// Map the features of each child to each parent.
	for i := 0; i < len(parents); i++ {
		srcFeature = append(srcFeature, childIPFeature...)
		srcNegFeature = append(srcNegFeature, childNegIPFeatures...)
		srcNegNegFeature = append(srcNegNegFeature, childNegNegIPFeatures...)
	}

	// Find the aggregation hosts for parents.
	parentsFirstOrderNeighbours := make([][]*resource.Host, 0, defaultAggregationNumber)
	parentsSecondOrderNeighbours := make([][][]*resource.Host, 0, defaultAggregationNumber)
	for _, parent := range parents {
		parentFirstOrderNeighbours, parentSecondOrderNeighbours, err := e.aggregationHosts(parent.Host, defaultAggregationNumber)
		if err != nil {
			logger.Error(err)
			return []float64{}, err
		}

		parentsFirstOrderNeighbours = append(parentsFirstOrderNeighbours, parentFirstOrderNeighbours)
		parentsSecondOrderNeighbours = append(parentsSecondOrderNeighbours, parentSecondOrderNeighbours)
	}

	// Generate feature vector for parents and theirs aggregation hosts.
	var (
		destFeature       = make([]float32, 0, len(parents)*defaultIPv4FeatureLength)
		destNegFeature    = make([]float32, 0, len(parents)*defaultNeighbourIpFeatureLength)
		destNegNegFeature = make([]float32, 0, len(parents)*defaultNeighbourNeighbourIpFeatureLength)
	)

	for i, parent := range parents {
		feature, err := parseIP(parent.Host.IP)
		if err != nil {
			return []float64{}, err
		}

		destFeature = append(destFeature, feature...)

		for _, parentFirstOrderNeighbours := range parentsFirstOrderNeighbours[i] {
			feature, err := parseIP(parentFirstOrderNeighbours.IP)
			if err != nil {
				return []float64{}, err
			}

			destNegFeature = append(destNegFeature, feature...)
		}

		for _, parentSecondOrderNeighbours := range parentsSecondOrderNeighbours[i] {
			for _, host := range parentSecondOrderNeighbours {
				feature, err := parseIP(host.IP)
				if err != nil {
					return []float64{}, err
				}

				destNegNegFeature = append(destNegNegFeature, feature...)
			}
		}
	}

	inferInputs := []*triton.ModelInferRequest_InferInputTensor{
		{
			Name:     "src",
			Datatype: "FP32",
			Shape:    []int64{int64(len(parents)), defaultIPv4FeatureLength},
			Contents: &triton.InferTensorContents{
				Fp32Contents: srcFeature,
			},
		},
		{
			Name:     "src_neg",
			Datatype: "FP32",
			Shape:    []int64{int64(len(parents)), defaultAggregationNumber, defaultIPv4FeatureLength},
			Contents: &triton.InferTensorContents{
				Fp32Contents: srcNegFeature,
			},
		},
		{
			Name:     "src_neg_neg",
			Datatype: "FP32",
			Shape:    []int64{int64(len(parents)), defaultAggregationNumber, defaultAggregationNumber, defaultIPv4FeatureLength},
			Contents: &triton.InferTensorContents{
				Fp32Contents: srcNegNegFeature,
			},
		},
		{
			Name:     "dst",
			Datatype: "FP32",
			Shape:    []int64{int64(len(parents)), defaultIPv4FeatureLength},
			Contents: &triton.InferTensorContents{
				Fp32Contents: destFeature,
			},
		},
		{
			Name:     "dst_neg",
			Datatype: "FP32",
			Shape:    []int64{int64(len(parents)), defaultAggregationNumber, defaultIPv4FeatureLength},
			Contents: &triton.InferTensorContents{
				Fp32Contents: destNegFeature,
			},
		},
		{
			Name:     "dst_neg_neg",
			Datatype: "FP32",
			Shape:    []int64{int64(len(parents)), defaultAggregationNumber, defaultAggregationNumber, defaultIPv4FeatureLength},
			Contents: &triton.InferTensorContents{
				Fp32Contents: destNegNegFeature,
			},
		},
	}

	inferOutputs := []*triton.ModelInferRequest_InferRequestedOutputTensor{
		{
			Name: "output_0",
		},
	}

	inferRequest := triton.ModelInferRequest{
		ModelName:    "models",
		ModelVersion: "1",
		Inputs:       inferInputs,
		Outputs:      inferOutputs,
	}

	inferResponse, err := e.inferenceClient.ModelInfer(context.Background(), &inferRequest)
	if err != nil {
		logger.Info(err)
		return []float64{}, err
	}

	data := postprocess(inferResponse.RawOutputContents)
	// Storage graphsage record.
	outputs := make([]float64, len(data))
	for i, v := range data {
		outputs[i] = float64(v)
		if err := e.storage.CreateGraphsage(storage.Graphsage{
			ID: child.ID,
			SrcHost: storage.GraphsageHost{
				ID:       child.Host.ID,
				Type:     child.Host.Type.Name(),
				Hostname: child.Host.Hostname,
				IP:       child.Host.IP,
				Port:     child.Host.Port,
				Network: resource.Network{
					TCPConnectionCount:       child.Host.Network.TCPConnectionCount,
					UploadTCPConnectionCount: child.Host.Network.UploadTCPConnectionCount,
					Location:                 child.Host.Network.Location,
					IDC:                      child.Host.Network.IDC,
				},
			},
			DestHost: storage.GraphsageHost{
				ID:       parents[i].Host.ID,
				Type:     parents[i].Host.Type.Name(),
				Hostname: parents[i].Host.Hostname,
				IP:       parents[i].Host.IP,
				Port:     parents[i].Host.Port,
				Network: resource.Network{
					TCPConnectionCount:       parents[i].Host.Network.TCPConnectionCount,
					UploadTCPConnectionCount: parents[i].Host.Network.UploadTCPConnectionCount,
					Location:                 parents[i].Host.Network.Location,
					IDC:                      parents[i].Host.Network.IDC,
				},
			},
			SrcFeature:        srcFeature,
			SrcNegFeature:     srcNegFeature,
			SrcNegNegFeature:  srcNegNegFeature,
			DestFeature:       destFeature,
			DestNegFeature:    destNegFeature,
			DestNegNegFeature: destNegNegFeature,
		}); err != nil {
			logger.Error(err)
			continue
		}
	}

	return outputs, nil
}

func (e *evaluatorMachineLearning) aggregationHosts(host *resource.Host, number int) ([]*resource.Host, [][]*resource.Host, error) {
	firstOrderNeighbours, err := e.networkTopology.Neighbours(host, number)
	if err != nil {
		return nil, nil, err
	}

	// If there is no neighbour host, using the root host as neighbour host.
	// If there is no enough neighbour host, randomly select neighbour host.
	var count = len(firstOrderNeighbours)
	if count == 0 {
		for i := 0; i < number; i++ {
			firstOrderNeighbours = append(firstOrderNeighbours, host)
		}
	} else if count < number {
		n := number - count
		for i := 0; i < n; i++ {
			firstOrderNeighbours = append(firstOrderNeighbours, firstOrderNeighbours[rand.Intn(count)])
		}
	}

	secondOrderNeighbours := make([][]*resource.Host, 0, number)
	for _, firstOrderNeighbour := range firstOrderNeighbours {
		neighbours, err := e.networkTopology.Neighbours(firstOrderNeighbour, number)
		if err != nil {
			return nil, nil, err
		}

		var count = len(neighbours)
		if count == 0 {
			for i := 0; i < number; i++ {
				neighbours = append(neighbours, firstOrderNeighbour)
			}
		} else if count < number {
			n := number - count
			for i := 0; i < n; i++ {
				neighbours = append(neighbours, neighbours[rand.Intn(count)])
			}
		}

		secondOrderNeighbours = append(secondOrderNeighbours, neighbours)
	}

	return firstOrderNeighbours, secondOrderNeighbours, nil
}

// Convert output's raw bytes into float32 data (Little Endian).
func postprocess(raw [][]byte) []float32 {
	outputs := make([]float32, len(raw[0])/sizeFloat32)
	for i := range outputs {
		offset := i * sizeFloat32
		s := raw[0][offset : offset+sizeFloat32]
		outputs[i] = byteToFloat32(s)
	}

	return outputs
}

// Convert byte to float32.
func byteToFloat32(v []byte) float32 {
	bits := binary.LittleEndian.Uint32(v)
	return math.Float32frombits(bits)
}

// parseIP parses an ip address to a feature vector.
func parseIP(ip string) ([]float32, error) {
	var features = make([]float32, 32)
	prase := net.ParseIP(ip).To4()
	if prase == nil {
		logger.Info(ip)
		return nil, errors.New("prase ip error")
	}

	for i := 0; i < net.IPv4len; i++ {
		d := prase[i]
		for j := 0; j < 8; j++ {
			features[i*8+j] = float32(d & 0x1)
			d = d >> 1
		}

	}

	return features, nil
}
