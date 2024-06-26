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

package job

type PreheatRequest struct {
	URL                 string            `json:"url" validate:"required,url"`
	Tag                 string            `json:"tag" validate:"omitempty"`
	Digest              string            `json:"digest" validate:"omitempty"`
	FilteredQueryParams string            `json:"filteredQueryParams" validate:"omitempty"`
	Headers             map[string]string `json:"headers" validate:"omitempty"`
	Application         string            `json:"application" validate:"omitempty"`
	Priority            int32             `json:"priority" validate:"omitempty"`
	PieceLength         uint32            `json:"pieceLength" validate:"omitempty"`
}

type PreheatResponse struct {
}
