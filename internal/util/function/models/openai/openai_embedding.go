// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"time"

	"github.com/milvus-io/milvus/internal/util/function/models/utils"
)

type EmbeddingRequest struct {
	// ID of the model to use.
	Model string `json:"model"`

	// Input text to embed, encoded as a string.
	Input []string `json:"input"`

	// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
	User string `json:"user,omitempty"`

	// The format to return the embeddings in. Can be either float or base64.
	EncodingFormat string `json:"encoding_format,omitempty"`

	// The number of dimensions the resulting output embeddings should have. Only supported in text-embedding-3 and later models.
	Dimensions int `json:"dimensions,omitempty"`
}

type Usage struct {
	// The number of tokens used by the prompt.
	PromptTokens int `json:"prompt_tokens"`

	// The total number of tokens used by the request.
	TotalTokens int `json:"total_tokens"`
}

type EmbeddingData struct {
	// The object type, which is always "embedding".
	Object string `json:"object"`

	// The embedding vector, which is a list of floats.
	Embedding []float32 `json:"embedding"`

	// The index of the embedding in the list of embeddings.
	Index int `json:"index"`
}

type EmbeddingResponse struct {
	// The object type, which is always "list".
	Object string `json:"object"`

	// The list of embeddings generated by the model.
	Data []EmbeddingData `json:"data"`

	// The name of the model used to generate the embedding.
	Model string `json:"model"`

	// The usage information for the request.
	Usage Usage `json:"usage"`
}

type ByIndex struct {
	resp *EmbeddingResponse
}

func (eb *ByIndex) Len() int { return len(eb.resp.Data) }
func (eb *ByIndex) Swap(i, j int) {
	eb.resp.Data[i], eb.resp.Data[j] = eb.resp.Data[j], eb.resp.Data[i]
}
func (eb *ByIndex) Less(i, j int) bool { return eb.resp.Data[i].Index < eb.resp.Data[j].Index }

type ErrorInfo struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Param   string `json:"param,omitempty"`
	Type    string `json:"type"`
}

type EmbedddingError struct {
	Error ErrorInfo `json:"error"`
}

type OpenAIEmbeddingInterface interface {
	Check() error
	Embedding(modelName string, texts []string, dim int, user string, timeoutSec int64) (*EmbeddingResponse, error)
}

type openAIBase struct {
	apiKey string
	url    string
}

func (c *openAIBase) Check() error {
	if c.apiKey == "" {
		return fmt.Errorf("api key is empty")
	}

	if c.url == "" {
		return fmt.Errorf("url is empty")
	}
	return nil
}

func (c *openAIBase) genReq(modelName string, texts []string, dim int, user string) *EmbeddingRequest {
	var r EmbeddingRequest
	r.Model = modelName
	r.Input = texts
	r.EncodingFormat = "float"
	if user != "" {
		r.User = user
	}
	if dim != 0 {
		r.Dimensions = dim
	}
	return &r
}

func (c *openAIBase) embedding(url string, headers map[string]string, modelName string, texts []string, dim int, user string, timeoutSec int64) (*EmbeddingResponse, error) {
	r := c.genReq(modelName, texts, dim, user)
	data, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}

	if timeoutSec <= 0 {
		timeoutSec = utils.DefaultTimeout
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeoutSec)*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	body, err := utils.RetrySend(req, 3)
	if err != nil {
		return nil, err
	}

	var res EmbeddingResponse
	err = json.Unmarshal(body, &res)
	if err != nil {
		return nil, err
	}
	sort.Sort(&ByIndex{&res})
	return &res, err
}

type OpenAIEmbeddingClient struct {
	openAIBase
}

func NewOpenAIEmbeddingClient(apiKey string, url string) *OpenAIEmbeddingClient {
	return &OpenAIEmbeddingClient{
		openAIBase{
			apiKey: apiKey,
			url:    url,
		},
	}
}

func (c *OpenAIEmbeddingClient) Embedding(modelName string, texts []string, dim int, user string, timeoutSec int64) (*EmbeddingResponse, error) {
	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": fmt.Sprintf("Bearer %s", c.apiKey),
	}
	return c.embedding(c.url, headers, modelName, texts, dim, user, timeoutSec)
}

type AzureOpenAIEmbeddingClient struct {
	openAIBase
	apiVersion string
}

func NewAzureOpenAIEmbeddingClient(apiKey string, url string) *AzureOpenAIEmbeddingClient {
	return &AzureOpenAIEmbeddingClient{
		openAIBase: openAIBase{
			apiKey: apiKey,
			url:    url,
		},
		apiVersion: "2024-06-01",
	}
}

func (c *AzureOpenAIEmbeddingClient) Embedding(modelName string, texts []string, dim int, user string, timeoutSec int64) (*EmbeddingResponse, error) {
	base, err := url.Parse(c.url)
	if err != nil {
		return nil, err
	}
	path := fmt.Sprintf("/openai/deployments/%s/embeddings", modelName)
	base.Path = path
	params := url.Values{}
	params.Add("api-version", c.apiVersion)
	base.RawQuery = params.Encode()
	url := base.String()

	headers := map[string]string{
		"Content-Type": "application/json",
		"api-key":      c.apiKey,
	}
	return c.embedding(url, headers, modelName, texts, dim, user, timeoutSec)
}
