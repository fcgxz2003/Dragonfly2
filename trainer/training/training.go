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
	"errors"
	"fmt"

	tf "github.com/galeone/tensorflow/tensorflow/go"
	"github.com/gocarina/gocsv"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	logger "d7y.io/dragonfly/v2/internal/dflog"
	"d7y.io/dragonfly/v2/pkg/idgen"
	pkgredis "d7y.io/dragonfly/v2/pkg/redis"
	managerclient "d7y.io/dragonfly/v2/pkg/rpc/manager/client"
	schedulerstorage "d7y.io/dragonfly/v2/scheduler/storage"
	"d7y.io/dragonfly/v2/trainer/config"
	"d7y.io/dragonfly/v2/trainer/storage"
)

//go:generate mockgen -destination mocks/training_mock.go -source training.go -package mocks

const (
	defaultAggregationNumber = 3

	defaultIPv4FeatureLength = 32

	defaultBatchSize = 5

	defaultEpoch = 1
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

	// Record Buffer.
	buffer []Record

	// Training epoch.
	epoch int

	// Loss of each epoch.
	losses []float32
}

// New returns a new Training.
func New(cfg *config.Config, managerClient managerclient.V2, storage storage.Storage) Training {
	return &training{
		config:        cfg,
		storage:       storage,
		managerClient: managerClient,
		buffer:        make([]Record, 0),
	}
}

// Train begins training GNN and MLP model.
func (t *training) Train(ctx context.Context, ip, hostname string) error {
	records, err := t.preprocess(ip, hostname)
	if err != nil {
		logger.Error(err)
		return err
	}

	// Write record to buffer.
	t.buffer = append(t.buffer, records...)
	for {
		if len(t.buffer) < defaultBatchSize {
			break
		}

		if err := t.train(t.buffer[:defaultBatchSize]); err != nil {
			logger.Info(err)
			continue
		}

		t.buffer = t.buffer[defaultBatchSize:]
	}

	var hostID = idgen.HostIDV2(ip, hostname)

	// Clean up download data.
	if err := t.storage.ClearDownload(hostID); err != nil {
		logger.Error(err)
		return err
	}

	// Clean up graphsage records data.
	if err := t.storage.ClearGraphsage(hostID); err != nil {
		logger.Error(err)
		return err
	}

	return nil
}

func (t *training) preprocess(ip, hostname string) ([]Record, error) {
	var hostID = idgen.HostIDV2(ip, hostname)
	// Preprocess download training data.
	logger.Info("loading download.csv")
	downloadFile, err := t.storage.OpenDownload(hostID)
	if err != nil {
		msg := fmt.Sprintf("open download failed: %s", err.Error())
		logger.Error(msg)
		return nil, status.Error(codes.Internal, msg)
	}
	defer downloadFile.Close()

	bandwidths := make(map[string]float32)
	dc := make(chan schedulerstorage.Download)
	go func() {
		if gocsv.UnmarshalToChanWithoutHeaders(downloadFile, dc) != nil {
			logger.Errorf("prase download file filed: %s", err.Error())
		}
	}()
	for download := range dc {
		logger.Info(download)
		for _, parent := range download.Parents {
			if parent.ID != "" {
				// get maxBandwidth locally from pieces.
				var localMaxBandwidth float32
				for _, piece := range parent.Pieces {
					if piece.Cost > 0 && float32(piece.Length/piece.Cost) > localMaxBandwidth {
						localMaxBandwidth = float32(piece.Length / piece.Cost)
					}
				}

				// updata maxBandwidth globally.
				key := pkgredis.MakeBandwidthKeyInTrainer(download.ID, download.Host.IP, parent.Host.IP)
				if value, ok := bandwidths[key]; ok {
					if localMaxBandwidth > value {
						bandwidths[key] = localMaxBandwidth
					}
				} else {
					bandwidths[key] = localMaxBandwidth
				}
			}
		}
	}

	for k, v := range bandwidths {
		logger.Info("%s:%s", k, v)
	}

	// Preprocess graphsage training data.
	logger.Info("loading graphsage.csv")
	graphsageFile, err := t.storage.OpenGraphsage(hostID)
	if err != nil {
		msg := fmt.Sprintf("open graphsage records failed: %s", err.Error())
		logger.Error(msg)
		return nil, status.Error(codes.Internal, msg)
	}
	defer graphsageFile.Close()

	records := make([]Record, 0)
	gc := make(chan schedulerstorage.Graphsage)
	go func() {
		if gocsv.UnmarshalToChanWithoutHeaders(graphsageFile, gc) != nil {
			logger.Errorf("prase graphsage file filed: %s", err.Error())
		}
	}()
	for graphsage := range gc {
		logger.Info(graphsage)
		key := pkgredis.MakeBandwidthKeyInTrainer(graphsage.ID, graphsage.SrcHost.IP, graphsage.DestHost.IP)
		if value, ok := bandwidths[key]; ok {
			record := Record{}
			record.Bandwidth = value

			// root.
			record.SrcFeature = graphsage.SrcFeature
			record.DestFeature = graphsage.DestFeature

			// neighbour.
			for i := 0; i < defaultAggregationNumber; i++ {
				start := i * defaultIPv4FeatureLength
				end := start + defaultIPv4FeatureLength
				record.SrcNegFeature = append(record.SrcNegFeature, graphsage.SrcNegFeature[start:end])
			}

			for i := 0; i < defaultAggregationNumber; i++ {
				start := i * defaultIPv4FeatureLength
				end := start + defaultIPv4FeatureLength
				record.DestNegFeature = append(record.DestNegFeature, graphsage.DestNegFeature[start:end])
			}

			// neighbour neighbour.
			for i := 0; i < defaultAggregationNumber; i++ {
				tmpSrcNegFeature := make([][]float32, 0, defaultAggregationNumber)
				for j := 0; j < defaultAggregationNumber; j++ {
					start := (i*defaultAggregationNumber + j) * defaultIPv4FeatureLength
					end := start + defaultIPv4FeatureLength
					tmpSrcNegFeature = append(tmpSrcNegFeature, graphsage.SrcNegNegFeature[start:end])
				}

				record.SrcNegNegFeature = append(record.SrcNegNegFeature, tmpSrcNegFeature)
			}

			for i := 0; i < defaultAggregationNumber; i++ {
				tmpDestNegFeature := make([][]float32, 0, defaultAggregationNumber)
				for j := 0; j < defaultAggregationNumber; j++ {
					start := (i*defaultAggregationNumber + j) * defaultIPv4FeatureLength
					end := start + defaultIPv4FeatureLength
					tmpDestNegFeature = append(tmpDestNegFeature, graphsage.DestNegNegFeature[start:end])
				}

				record.DestNegNegFeature = append(record.DestNegNegFeature, tmpDestNegFeature)
			}

			records = append(records, record)
		}
	}

	logger.Info(records)
	return records, nil
}

func (t *training) train(records []Record) error {
	gm, _ := tf.LoadSavedModel("base_model", []string{"serve"}, nil)

	// Reach the training rounds, save and upload the model.
	if t.epoch >= defaultEpoch {
		// TODO
		logger.Info("save model")
		t.epoch = 0
		t.losses = t.losses[:0]
	}

	var (
		srcRawData       = make([][]float32, 0, defaultBatchSize)
		srcNegRawData    = make([][][]float32, 0, defaultBatchSize)
		srcNegNegRawData = make([][][][]float32, 0, defaultBatchSize)
		dstRawData       = make([][]float32, 0, defaultBatchSize)
		dstNegRawData    = make([][][]float32, 0, defaultBatchSize)
		dstNegNegRawData = make([][][][]float32, 0, defaultBatchSize)
		labelsRawData    = make([]float32, 0, defaultBatchSize)
	)

	for _, record := range records {
		srcRawData = append(srcRawData, record.SrcFeature)
		srcNegRawData = append(srcNegRawData, record.SrcNegFeature)
		srcNegNegRawData = append(srcNegNegRawData, record.SrcNegNegFeature)
		dstRawData = append(dstRawData, record.DestFeature)
		dstNegRawData = append(dstNegRawData, record.DestNegFeature)
		dstNegNegRawData = append(dstNegNegRawData, record.DestNegNegFeature)
		labelsRawData = append(labelsRawData, record.Bandwidth)
	}

	// Convert raw data to tensor.
	src, err := tf.NewTensor(srcRawData)
	if err != nil {
		logger.Error(err)
		return err
	}

	srcNeg, err := tf.NewTensor(srcNegRawData)
	if err != nil {
		logger.Error(err)
		return err
	}

	srcNegNeg, err := tf.NewTensor(srcNegNegRawData)
	if err != nil {
		logger.Error(err)
		return err
	}

	dst, err := tf.NewTensor(dstRawData)
	if err != nil {
		logger.Error(err)
		return err
	}

	dstNeg, err := tf.NewTensor(dstNegRawData)
	if err != nil {
		logger.Error(err)
		return err
	}

	dstNegNeg, err := tf.NewTensor(dstNegNegRawData)
	if err != nil {
		logger.Error(err)
		return err
	}

	labels, err := tf.NewTensor(labelsRawData)
	if err != nil {
		logger.Error(err)
		return err
	}

	// Start training.
	result, err := gm.Session.Run(
		map[tf.Output]*tf.Tensor{
			gm.Graph.Operation("train_src").Output(0):         src,
			gm.Graph.Operation("train_src_neg").Output(0):     srcNeg,
			gm.Graph.Operation("train_src_neg_neg").Output(0): srcNegNeg,
			gm.Graph.Operation("train_dst").Output(0):         dst,
			gm.Graph.Operation("train_dst_neg").Output(0):     dstNeg,
			gm.Graph.Operation("train_dst_neg_neg").Output(0): dstNegNeg,
			gm.Graph.Operation("train_labels").Output(0):      labels,
		},
		[]tf.Output{
			gm.Graph.Operation("StatefulPartitionedCall_1").Output(0),
		},
		nil,
	)
	if err != nil {
		logger.Error(err)
		return err
	}

	loss, ok := result[0].Value().(float32)
	if !ok {
		logger.Info("error output")
		return errors.New("error output")
	}

	t.losses = append(t.losses, loss)
	t.epoch++
	return nil
}
