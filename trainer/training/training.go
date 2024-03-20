/*
 *     Copyright 2023 The Dragonfly Authors
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

package training

import (
	"context"
	"fmt"

	"github.com/gocarina/gocsv"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	logger "d7y.io/dragonfly/v2/internal/dflog"
	"d7y.io/dragonfly/v2/pkg/idgen"
	managerclient "d7y.io/dragonfly/v2/pkg/rpc/manager/client"
	schedulerstorage "d7y.io/dragonfly/v2/scheduler/storage"
	"d7y.io/dragonfly/v2/trainer/config"
	"d7y.io/dragonfly/v2/trainer/storage"
)

//go:generate mockgen -destination mocks/training_mock.go -source training.go -package mocks

const (
	defaultAggregationNumber = 3

	defaultIPv4FeatureLength = 32

	// BandwidthNamespace prefix of bandwidth namespace cache key.
	BandwidthNamespace = "bandwidth"

	// TrainerName is name of trainer.
	TrainerName = "trainer"
)

// Training defines the interface to train GNN and MLP model.
type Training interface {
	// Train begins training GNN and MLP model.
	Train(context.Context, string, string) error
}

// training implements Training interface.
type training struct {
	// Trainer service config.
	config *config.Config

	// Storage interface.
	storage storage.Storage

	// Manager service clent.
	managerClient managerclient.V2
}

// New returns a new Training.
func New(cfg *config.Config, managerClient managerclient.V2, storage storage.Storage) Training {
	return &training{
		config:        cfg,
		storage:       storage,
		managerClient: managerClient,
	}
}

// Train begins training GNN and MLP model.
func (t *training) Train(ctx context.Context, ip, hostname string) error {
	if err := t.preprocess(ip, hostname); err != nil {
		logger.Error(err)
		return err
	}

	// Clean up training data.
	if err := t.storage.Clear(); err != nil {
		logger.Error(err)
		return err
	}

	return nil
}

func (t *training) preprocess(ip, hostname string) error {
	var hostID = idgen.HostIDV2(ip, hostname)
	// Preprocess download training data.
	downloadFile, err := t.storage.OpenDownload(hostID)
	if err != nil {
		msg := fmt.Sprintf("open download failed: %s", err.Error())
		logger.Error(msg)
		return status.Error(codes.Internal, msg)
	}
	defer downloadFile.Close()

	// bandwidths := make(map[string]float64)
	dc := make(chan schedulerstorage.Download)
	go func() {
		if gocsv.UnmarshalToChanWithoutHeaders(downloadFile, dc) != nil {
			logger.Errorf("prase download file filed: %s", err.Error())
		}
	}()
	for download := range dc {
		logger.Info(download)
		// for _, parent := range download.Parents {
		// 	if parent.ID != "" {
		// 		// get maxBandwidth locally from pieces.
		// 		var localMaxBandwidth float64
		// 		for _, piece := range parent.Pieces {
		// 			if piece.Cost > 0 && float64(piece.Length/piece.Cost) > localMaxBandwidth {
		// 				localMaxBandwidth = float64(piece.Length / piece.Cost)
		// 			}
		// 		}

		// 		// updata maxBandwidth globally.
		// 		key := makeBandwidthKeyInTrainer(download.Host.IP, parent.Host.IP)
		// 		if value, ok := bandwidths[key]; ok {
		// 			if localMaxBandwidth > value {
		// 				bandwidths[key] = localMaxBandwidth
		// 			}
		// 		} else {
		// 			bandwidths[key] = localMaxBandwidth
		// 		}
		// 	}
		// }
	}

	// Preprocess graphsage training data.
	graphsageFile, err := t.storage.OpenGraphsage(hostID)
	if err != nil {
		msg := fmt.Sprintf("open graphsage records failed: %s", err.Error())
		logger.Error(msg)
		return status.Error(codes.Internal, msg)
	}
	defer graphsageFile.Close()

	// records := Records{}
	gc := make(chan schedulerstorage.Graphsage)
	go func() {
		if gocsv.UnmarshalToChanWithoutHeaders(graphsageFile, gc) != nil {
			logger.Errorf("prase graphsage file filed: %s", err.Error())
		}
	}()
	for graphsage := range gc {
		logger.Info(graphsage)
		// // root.
		// records.SrcFeature = append(records.SrcFeature, graphsage.SrcFeature)
		// records.DestFeature = append(records.DestFeature, graphsage.DestFeature)

		// // neighbour.
		// tmpSrcNegFeature := make([][]float32, 0, defaultAggregationNumber)
		// for i := 0; i < defaultAggregationNumber; i++ {
		// 	start := i * defaultIPv4FeatureLength
		// 	end := start + defaultIPv4FeatureLength
		// 	tmpSrcNegFeature = append(tmpSrcNegFeature, graphsage.SrcNegFeature[start:end])
		// }
		// records.SrcNegFeature = append(records.SrcNegFeature, tmpSrcNegFeature)

		// tmpDestNegFeature := make([][]float32, 0, defaultAggregationNumber)
		// for i := 0; i < defaultAggregationNumber; i++ {
		// 	start := i * defaultIPv4FeatureLength
		// 	end := start + defaultIPv4FeatureLength
		// 	tmpDestNegFeature = append(tmpDestNegFeature, graphsage.DestFeature[start:end])
		// }
		// records.DestNegFeature = append(records.SrcNegFeature, tmpDestNegFeature)

		// // neighbour neighbour.
		// tmpSrcNegNegFeature := make([][][]float32, 0, defaultAggregationNumber)
		// for i := 0; i < defaultAggregationNumber; i++ {
		// 	tmpSrcNegFeature := make([][]float32, 0, defaultAggregationNumber)
		// 	for j := 0; j < defaultAggregationNumber; j++ {
		// 		start := (i*defaultAggregationNumber + j) * defaultIPv4FeatureLength
		// 		end := start + defaultIPv4FeatureLength
		// 		tmpSrcNegFeature = append(tmpSrcNegFeature, graphsage.SrcNegNegFeature[start:end])
		// 	}

		// 	tmpSrcNegNegFeature = append(tmpSrcNegNegFeature, tmpSrcNegFeature)
		// }
		// records.SrcNegNegFeature = append(records.SrcNegNegFeature, tmpSrcNegNegFeature)

		// tmpDestNegNegFeature := make([][][]float32, 0, defaultAggregationNumber)
		// for i := 0; i < defaultAggregationNumber; i++ {
		// 	tmpDestNegFeature := make([][]float32, 0, defaultAggregationNumber)
		// 	for j := 0; j < defaultAggregationNumber; j++ {
		// 		start := (i*defaultAggregationNumber + j) * defaultIPv4FeatureLength
		// 		end := start + defaultIPv4FeatureLength
		// 		tmpDestNegFeature = append(tmpDestNegFeature, graphsage.DestNegNegFeature[start:end])
		// 	}

		// 	tmpDestNegNegFeature = append(tmpDestNegNegFeature, tmpDestNegFeature)
		// }
		// records.DestNegNegFeature = append(records.DestNegNegFeature, tmpDestNegNegFeature)
	}

	return nil
}

// makeNamespaceKeyInTrainer make namespace key in trainer.
func makeNamespaceKeyInTrainer(namespace string) string {
	return fmt.Sprintf("%s:%s", TrainerName, namespace)
}

// makeKeyInTrainer make key in trainer.
func makeKeyInTrainer(namespace, id string) string {
	return fmt.Sprintf("%s:%s", makeNamespaceKeyInTrainer(namespace), id)
}

// makeBandwidthKeyInTrainer make bandwidth key in trainer.
func makeBandwidthKeyInTrainer(srcHostID, destHostID string) string {
	return makeKeyInTrainer(BandwidthNamespace, fmt.Sprintf("%s:%s", srcHostID, destHostID))
}
