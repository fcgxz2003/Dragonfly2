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
	"io"
	"os"
	"path/filepath"
	"time"

	tf "github.com/galeone/tensorflow/tensorflow/go"
	"github.com/gocarina/gocsv"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
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

	BucketName = "models"
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

	baseDir string

	// Storage interface.
	storage storage.Storage

	// Manager service clent.
	managerClient managerclient.V1

	minioClient *minio.Client

	// Record Buffer.
	buffer []Record

	// Training epoch.
	epoch int
}

// New returns a new Training.
func New(cfg *config.Config, baseDir string, managerClient managerclient.V1, storage storage.Storage) (Training, error) {
	// Initialize minio client object.
	minioClient, err := minio.New(cfg.Minio.Endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4("admin", "admin123", ""),
		Secure: false,
	})
	if err != nil {
		return nil, err
	}

	ctx := context.Background()
	exists, err := minioClient.BucketExists(ctx, BucketName)
	if err == nil && !exists {
		if err := minioClient.MakeBucket(ctx, BucketName, minio.MakeBucketOptions{}); err != nil {
			logger.Info(err)
			return nil, err
		}

		baseModelPath := fmt.Sprintf("%s/%s", baseDir, "base_model")
		if err := uploadBaseModel(minioClient, baseModelPath); err != nil {
			logger.Info(err)
			return nil, err
		}
	} else if err != nil {
		logger.Info(err)
		return nil, err
	}

	return &training{
		config:        cfg,
		baseDir:       baseDir,
		storage:       storage,
		managerClient: managerClient,
		minioClient:   minioClient,
		buffer:        make([]Record, 0),
	}, nil
}

// Train begins training GNN and MLP model.
func (t *training) Train(ctx context.Context, ip, hostname string) error {
	records, err := t.preprocess(ip, hostname)
	if err != nil {
		logger.Error(err)
		return err
	}

	// Write record to buffer.
	batchsize := t.config.Train.BatchSize
	t.buffer = append(t.buffer, records...)
	for {
		if len(t.buffer) < batchsize {
			break
		}

		trainData := t.buffer[:batchsize]
		t.buffer = t.buffer[batchsize:]
		if err := t.train(trainData, ip, hostname); err != nil {
			logger.Info(err)
			continue
		}

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
		for _, parent := range download.Parents {
			if parent.ID != "" {
				// get maxBandwidth locally from pieces.
				var localMaxBandwidth float32
				for _, piece := range parent.Pieces {
					bandwidth := float32(float64(piece.Length) / 1024 / time.Duration(piece.Cost).Seconds())
					if piece.Cost > 0 && bandwidth > localMaxBandwidth {
						localMaxBandwidth = bandwidth
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

	// Preprocess graphsage training data.
	logger.Info("loading graphsage.csv")
	graphsageFile, err := t.storage.OpenGraphsage(hostID)
	if err != nil {
		msg := fmt.Sprintf("open graphsage records failed: %s", err.Error())
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

	return records, nil
}

func (t *training) train(records []Record, ip, hostname string) error {
	modelPath := fmt.Sprintf("%s/%s:%s", t.baseDir, ip, hostname)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		// copy base model to this scheduler model.
		baseModelPath := fmt.Sprintf("%s/%s", t.baseDir, "base_model")
		if err := copyFolder(baseModelPath, modelPath); err != nil {
			return err
		}
	}

	gm, err := tf.LoadSavedModel(fmt.Sprintf("%s%s", modelPath, "/1/model.savedmodel/"), []string{"serve"}, nil)
	if err != nil {
		return err
	}

	var (
		batchsize        = t.config.Train.BatchSize
		srcRawData       = make([][]float32, 0, batchsize)
		srcNegRawData    = make([][][]float32, 0, batchsize)
		srcNegNegRawData = make([][][][]float32, 0, batchsize)
		dstRawData       = make([][]float32, 0, batchsize)
		dstNegRawData    = make([][][]float32, 0, batchsize)
		dstNegNegRawData = make([][][][]float32, 0, batchsize)
		labelsRawData    = make([]float32, 0, batchsize)
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
		return err
	}

	srcNeg, err := tf.NewTensor(srcNegRawData)
	if err != nil {
		return err
	}

	srcNegNeg, err := tf.NewTensor(srcNegNegRawData)
	if err != nil {
		return err
	}

	dst, err := tf.NewTensor(dstRawData)
	if err != nil {
		return err
	}

	dstNeg, err := tf.NewTensor(dstNegRawData)
	if err != nil {
		return err
	}

	dstNegNeg, err := tf.NewTensor(dstNegNegRawData)
	if err != nil {
		return err
	}

	labels, err := tf.NewTensor(labelsRawData)
	if err != nil {
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
		return err
	}

	loss, ok := result[0].Value().(float32)
	if !ok {
		return errors.New("error loss")
	}

	logger.Infof("model train loss is: %f", loss)
	if err := t.saveModel(gm, ip, hostname); err != nil {
		return err
	}

	t.epoch++
	// Reach the training rounds, save and upload the model.
	if t.epoch >= t.config.Train.Epoch {
		t.epoch = 0
		if err := t.uploadModel(ip, hostname); err != nil {
			return err
		}

	}

	return nil
}

// Save tensorflow model.
func (t *training) saveModel(gm *tf.SavedModel, ip, hostname string) error {
	modelPath := fmt.Sprintf("%s/%s:%s", t.baseDir, ip, hostname)
	savedModel, err := os.ReadFile(fmt.Sprintf("%s%s", modelPath, "/1/model.savedmodel/saved_model.pb"))
	if err != nil {
		return err
	}

	configpbtxt, err := os.ReadFile(fmt.Sprintf("%s%s", modelPath, "/config.pbtxt"))
	if err != nil {
		return err
	}

	os.RemoveAll(modelPath)
	os.MkdirAll(fmt.Sprintf("%s%s", modelPath, "/1/model.savedmodel/variables"), os.ModePerm)
	if err := os.WriteFile(fmt.Sprintf("%s%s", modelPath, "/1/model.savedmodel/saved_model.pb"), savedModel, os.ModePerm); err != nil {
		return err
	}

	if err := os.WriteFile(fmt.Sprintf("%s%s", modelPath, "/config.pbtxt"), configpbtxt, os.ModePerm); err != nil {
		return err
	}

	fileName, err := tf.NewTensor(fmt.Sprintf("%s%s", modelPath, "/1/model.savedmodel/variables/variables"))
	if err != nil {
		return err
	}

	_, err = gm.Session.Run(
		map[tf.Output]*tf.Tensor{
			gm.Graph.Operation("saver_filename").Output(0): fileName,
		},
		[]tf.Output{
			gm.Graph.Operation("StatefulPartitionedCall_2").Output(0),
		},
		nil,
	)
	if err != nil {
		return err
	}

	logger.Info("save model success")
	return nil
}

// Upload base model to minio.
func uploadBaseModel(minioClient *minio.Client, baseModelPath string) error {
	ctx := context.Background()

	objectName := fmt.Sprintf("%s%s", baseModelPath, "/1/model.savedmodel/saved_model.pb")
	filePath := fmt.Sprintf("%s%s", baseModelPath, "/1/model.savedmodel/saved_model.pb")
	info, err := minioClient.FPutObject(ctx, BucketName, objectName, filePath, minio.PutObjectOptions{})
	if err != nil {
		return err
	}
	logger.Infof("Successfully uploaded %s of size %d\n", objectName, info.Size)

	objectName = fmt.Sprintf("%s%s", baseModelPath, "/config.pbtxt")
	filePath = fmt.Sprintf("%s%s", baseModelPath, "/config.pbtxt")
	info, err = minioClient.FPutObject(ctx, BucketName, objectName, filePath, minio.PutObjectOptions{})
	if err != nil {
		return err
	}
	logger.Infof("Successfully uploaded %s of size %d\n", objectName, info.Size)

	objectName = fmt.Sprintf("%s%s", baseModelPath, "/1/model.savedmodel/variables/variables.data-00000-of-00001")
	filePath = fmt.Sprintf("%s%s", baseModelPath, "/1/model.savedmodel/variables/variables.data-00000-of-00001")
	info, err = minioClient.FPutObject(ctx, BucketName, objectName, filePath, minio.PutObjectOptions{})
	if err != nil {
		return err
	}
	logger.Infof("Successfully uploaded %s of size %d\n", objectName, info.Size)

	objectName = fmt.Sprintf("%s%s", baseModelPath, "/1/model.savedmodel/variables/variables.index")
	filePath = fmt.Sprintf("%s%s", baseModelPath, "/1/model.savedmodel/variables/variables.index")
	info, err = minioClient.FPutObject(ctx, BucketName, objectName, filePath, minio.PutObjectOptions{})
	if err != nil {
		return err
	}
	logger.Infof("Successfully uploaded %s of size %d\n", objectName, info.Size)

	logger.Info("upload base model success")
	return nil
}

// Upload model to minio.
func (t *training) uploadModel(ip, hostname string) error {
	ctx := context.Background()
	modelPath := fmt.Sprintf("%s/%s:%s", t.baseDir, ip, hostname)

	objectName := fmt.Sprintf("%s:%s%s", ip, hostname, "/1/model.savedmodel/saved_model.pb")
	filePath := fmt.Sprintf("%s%s", modelPath, "/1/model.savedmodel/saved_model.pb")
	info, err := t.minioClient.FPutObject(ctx, BucketName, objectName, filePath, minio.PutObjectOptions{})
	if err != nil {
		return err
	}
	logger.Infof("Successfully uploaded %s of size %d\n", objectName, info.Size)

	objectName = fmt.Sprintf("%s:%s%s", ip, hostname, "/config.pbtxt")
	filePath = fmt.Sprintf("%s%s", modelPath, "/config.pbtxt")
	info, err = t.minioClient.FPutObject(ctx, BucketName, objectName, filePath, minio.PutObjectOptions{})
	if err != nil {
		return err
	}
	logger.Infof("Successfully uploaded %s of size %d\n", objectName, info.Size)

	objectName = fmt.Sprintf("%s:%s%s", ip, hostname, "/1/model.savedmodel/variables/variables.data-00000-of-00001")
	filePath = fmt.Sprintf("%s%s", modelPath, "/1/model.savedmodel/variables/variables.data-00000-of-00001")
	info, err = t.minioClient.FPutObject(ctx, BucketName, objectName, filePath, minio.PutObjectOptions{})
	if err != nil {
		return err
	}
	logger.Infof("Successfully uploaded %s of size %d\n", objectName, info.Size)

	objectName = fmt.Sprintf("%s:%s%s", ip, hostname, "/1/model.savedmodel/variables/variables.index")
	filePath = fmt.Sprintf("%s%s", modelPath, "/1/model.savedmodel/variables/variables.index")
	info, err = t.minioClient.FPutObject(ctx, BucketName, objectName, filePath, minio.PutObjectOptions{})
	if err != nil {
		return err
	}
	logger.Infof("Successfully uploaded %s of size %d\n", objectName, info.Size)

	logger.Info("upload model success")
	return nil
}

// copyFile copies file from source to target.
func copyFile(source, target string) error {
	sourceFile, err := os.Open(source)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	targetFile, err := os.Create(target)
	if err != nil {
		return err
	}
	defer targetFile.Close()

	if _, err = io.Copy(targetFile, sourceFile); err != nil {
		return err
	}

	return nil
}

// copyFolder copies file directory from source target.
func copyFolder(source, target string) error {
	sourceInfo, err := os.Stat(source)
	if err != nil {
		return err
	}

	if err = os.MkdirAll(target, sourceInfo.Mode()); err != nil {
		return err
	}

	entries, err := os.ReadDir(source)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		sourcePath := filepath.Join(source, entry.Name())
		targetPath := filepath.Join(target, entry.Name())

		if entry.IsDir() {
			if err = copyFolder(sourcePath, targetPath); err != nil {
				return err
			}
		} else {
			if err = copyFile(sourcePath, targetPath); err != nil {
				return err
			}
		}
	}

	return nil
}
