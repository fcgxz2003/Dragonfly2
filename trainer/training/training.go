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

	managerv1 "github.com/fcgxz2003/api/v2/pkg/apis/manager/v1"
	"golang.org/x/sync/errgroup"

	logger "d7y.io/dragonfly/v2/internal/dflog"
	managerclient "d7y.io/dragonfly/v2/pkg/rpc/manager/client"
	"d7y.io/dragonfly/v2/trainer/config"
	"d7y.io/dragonfly/v2/trainer/storage"
)

//go:generate mockgen -destination mocks/training_mock.go -source training.go -package mocks

// Training defines the interface to train GNN and MLP model.
type Training interface {
	// Start starts the training service.
	Start()

	// Stop stops the training service.
	Stop()

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
	managerClient managerclient.V1
}

// New returns a new Training.
func New(cfg *config.Config, managerClient managerclient.V1, storage storage.Storage) Training {
	return &training{
		config:        cfg,
		storage:       storage,
		managerClient: managerClient,
	}
}

// Start starts the training service by uploading base model for each scheduler.
func (t *training) Start() {
	// Compress base model to bytes.
	data, err := compress(GraphsageBaseModel)
	if err != nil {
		logger.Error("compress graphsage base model error: %v", err.Error())
		return
	}

	// Get all schedulers configuration.
	getSchedulersResp, err := t.managerClient.GetSchedulers(context.Background(), &managerv1.GetSchedulersRequest{
		SourceType: managerv1.SourceType_TRAINER_SOURCE,
	})
	if err != nil {
		logger.Error("get scheduler configuration error: %v", err.Error())
	}

	// Upload all models to s3 as base model for each scheduler.
	// The recall, precision and fiscore of base model depend on pre-trained data.
	for _, scheduler := range getSchedulersResp.Schedulers {
		if err := t.managerClient.CreateModel(context.Background(), &managerv1.CreateModelRequest{
			Hostname: scheduler.Hostname,
			Ip:       scheduler.Ip,
			Request: &managerv1.CreateModelRequest_CreateGnnRequest{
				CreateGnnRequest: &managerv1.CreateGNNRequest{
					Data:      data,
					Recall:    0,
					Precision: 0,
					F1Score:   0,
				},
			},
		}); err != nil {
			logger.Error("upload base model of scheudler %s to s3 error: %v", scheduler.Ip, err.Error())
		}
	}
}

// Stop stops the training service by delete all model in s3.
func (t *training) Stop() {
	// TDOO: delete all model.
}

// Train begins training GNN and MLP model.
func (t *training) Train(ctx context.Context, ip, hostname string) error {
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		return t.trainGNN(ctx, ip, hostname)
	})

	eg.Go(func() error {
		return t.trainMLP(ctx, ip, hostname)
	})

	// Wait for all train tasks to complete.
	if err := eg.Wait(); err != nil {
		logger.Errorf("training failed: %v", err)
		return err
	}

	// TODO Clean up training data.
	return nil
}

// TODO Add training GNN logic.
// trainGNN trains GNN model.
func (t *training) trainGNN(ctx context.Context, ip, hostname string) error {
	// 1. Get training data from storage.
	// 2. Preprocess training data.
	// 2. Train GNN model.
	// 3. Upload GNN model to manager service.
	return nil
}

// TODO Add training MLP logic.
// trainMLP trains MLP model.
func (t *training) trainMLP(ctx context.Context, ip, hostname string) error {
	// 1. Get training data from storage.
	// 2. Preprocess training data.
	// 2. Train MLP model.
	// 3. Upload MLP model to manager service.
	return nil
}
