package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
	"github.com/Ingenimax/agent-sdk-go/pkg/llm"
	"github.com/Ingenimax/agent-sdk-go/pkg/logging"
	"github.com/Ingenimax/agent-sdk-go/pkg/multitenancy"
	"github.com/Ingenimax/agent-sdk-go/pkg/retry"
)

// AnthropicClient implements the LLM interface for Anthropic
type AnthropicClient struct {
	APIKey              string
	Model               string
	BaseURL             string
	HTTPClient          *http.Client
	logger              logging.Logger
	retryExecutor       *retry.Executor
	vertexRetryExecutor *VertexRetryExecutor
	VertexConfig        *VertexConfig
}

// Option represents an option for configuring the Anthropic client
type Option func(*AnthropicClient)

// WithModel sets the model for the Anthropic client
func WithModel(model string) Option {
	return func(c *AnthropicClient) {
		c.Model = model
	}
}

// WithLogger sets the logger for the Anthropic client
func WithLogger(logger logging.Logger) Option {
	return func(c *AnthropicClient) {
		c.logger = logger
	}
}

// WithRetry configures retry policy for the client
func WithRetry(opts ...retry.Option) Option {
	return func(c *AnthropicClient) {
		ctx := context.Background()
		policy := retry.NewPolicy(opts...)

		c.logger.Debug(ctx, "Configuring retry", map[string]interface{}{
			"vertex_config_enabled": c.VertexConfig != nil && c.VertexConfig.Enabled,
			"vertex_config_region": func() string {
				if c.VertexConfig != nil {
					return c.VertexConfig.Region
				}
				return ""
			}(),
			"max_attempts": policy.MaximumAttempts,
		})

		if c.VertexConfig != nil && c.VertexConfig.Enabled {
			vertexPolicy := &Policy{
				InitialInterval:    policy.InitialInterval,
				BackoffCoefficient: policy.BackoffCoefficient,
				MaximumInterval:    policy.MaximumInterval,
				MaximumAttempts:    policy.MaximumAttempts,
			}
			c.vertexRetryExecutor = NewVertexRetryExecutor(c.VertexConfig, vertexPolicy)
			c.logger.Info(ctx, "Created vertex retry executor with multi-region support", map[string]interface{}{
				"region":       c.VertexConfig.Region,
				"max_attempts": policy.MaximumAttempts,
			})
		} else {
			c.retryExecutor = retry.NewExecutor(policy)
			c.logger.Info(ctx, "Created standard retry executor", map[string]interface{}{
				"max_attempts":   policy.MaximumAttempts,
				"vertex_enabled": false,
			})
		}
	}
}

// WithBaseURL sets the base URL for the Anthropic API
func WithBaseURL(baseURL string) Option {
	return func(c *AnthropicClient) {
		c.BaseURL = baseURL
	}
}

// WithHTTPClient sets the HTTP client for the Anthropic client
func WithHTTPClient(httpClient *http.Client) Option {
	return func(c *AnthropicClient) {
		c.HTTPClient = httpClient
	}
}

// WithVertexAI configures the client for Google Vertex AI
func WithVertexAI(region, projectID string) Option {
	return func(c *AnthropicClient) {
		ctx := context.Background()

		c.logger.Debug(ctx, "Configuring Vertex AI", map[string]interface{}{
			"region":                region,
			"projectID":             projectID,
			"retry_executor_exists": c.retryExecutor != nil,
		})

		vertexConfig, err := NewVertexConfig(ctx, region, projectID)
		if err != nil {
			c.logger.Error(ctx, "Failed to configure Vertex AI", map[string]interface{}{
				"error":     err.Error(),
				"region":    region,
				"projectID": projectID,
			})
			return
		}
		c.VertexConfig = vertexConfig
		c.BaseURL = vertexConfig.GetBaseURL()

		// If retry executor already exists, create vertex retry executor now
		if c.retryExecutor != nil {
			c.logger.Debug(ctx, "Creating vertex retry executor (retry executor exists)", map[string]interface{}{
				"region": region,
			})
			// Note: We need to extract the retry policy from the existing executor
			// For now, we'll create a default policy - this should be improved
			policy := &Policy{
				InitialInterval:    time.Second,
				BackoffCoefficient: 2.0,
				MaximumInterval:    time.Second * 30,
				MaximumAttempts:    3,
			}
			c.vertexRetryExecutor = NewVertexRetryExecutor(c.VertexConfig, policy)
			c.logger.Info(ctx, "Created vertex retry executor with multi-region support", map[string]interface{}{
				"region": region,
			})
		} else {
			c.logger.Debug(ctx, "Retry executor not yet configured, vertex retry executor will be created when retry is configured", nil)
		}

		c.logger.Info(ctx, "Configured client for Vertex AI", map[string]interface{}{
			"region":                        region,
			"projectID":                     projectID,
			"baseURL":                       c.BaseURL,
			"vertex_retry_executor_created": c.vertexRetryExecutor != nil,
		})
	}
}

// WithVertexAICredentials configures Vertex AI with explicit credentials
func WithVertexAICredentials(region, projectID, credentialsPath string) Option {
	return func(c *AnthropicClient) {
		ctx := context.Background()
		vertexConfig, err := NewVertexConfigWithCredentials(ctx, region, projectID, credentialsPath)
		if err != nil {
			c.logger.Error(ctx, "Failed to configure Vertex AI with credentials", map[string]interface{}{
				"error":           err.Error(),
				"region":          region,
				"projectID":       projectID,
				"credentialsPath": credentialsPath,
			})
			return
		}
		c.VertexConfig = vertexConfig
		c.BaseURL = vertexConfig.GetBaseURL()
		c.logger.Info(ctx, "Configured client for Vertex AI with credentials", map[string]interface{}{
			"region":          region,
			"projectID":       projectID,
			"credentialsPath": credentialsPath,
			"baseURL":         c.BaseURL,
		})
	}
}

// NewClient creates a new Anthropic client
func NewClient(apiKey string, options ...Option) *AnthropicClient {
	// Create client with default options
	client := &AnthropicClient{
		APIKey:     apiKey,
		Model:      Claude37Sonnet,
		BaseURL:    "https://api.anthropic.com",
		HTTPClient: &http.Client{Timeout: 60 * time.Second},
		logger:     logging.New(),
	}

	// Apply options
	for _, option := range options {
		option(client)
	}

	// After all options are applied, if we have both VertexConfig and retry policy but no vertex executor,
	// create the vertex retry executor now
	if client.VertexConfig != nil && client.VertexConfig.Enabled && client.retryExecutor != nil && client.vertexRetryExecutor == nil {
		// Extract policy from the regular executor (this is a workaround)
		// Since we can't access the policy directly, we'll need to recreate it
		// For now, we'll just log this situation
		client.logger.Error(context.TODO(), "Vertex AI configured with retry but vertex executor not created. This indicates option ordering issue - WithRetry should come after WithVertexAI.", map[string]interface{}{
			"vertex_config_enabled":        true,
			"retry_executor_exists":        true,
			"vertex_retry_executor_exists": false,
		})
	}

	// Log warning if model is not specified
	if client.Model == "" {
		client.logger.Warn(context.TODO(), "No model specified, model must be explicitly set with WithModel", nil)
	}

	return client
}

// ModelName constants for supported Anthropic models
const (
	// Claude 3 family
	Claude35Haiku  = "claude-3-5-haiku-latest"
	Claude35Sonnet = "claude-3-5-sonnet-latest"
	Claude3Opus    = "claude-3-opus-latest"
	Claude37Sonnet = "claude-3-7-sonnet-20250219" // Supports thinking tokens
	ClaudeSonnet4  = "claude-sonnet-4-20250514"   // Latest model with thinking
	ClaudeOpus4    = "claude-opus-4-20250514"     // Latest Opus with thinking
	ClaudeOpus41   = "claude-opus-4-1-20250805"   // Latest Opus 4.1
)

// SupportsThinking returns true if the model supports thinking tokens
func SupportsThinking(model string) bool {
	supportedModels := []string{
		"claude-3-7-sonnet-20250219",
		"claude-sonnet-4-20250514",
		"claude-opus-4-20250514",
		"claude-opus-4-1-20250805",
		"claude-sonnet-4-5-20250929",
		// Vertex AI format models
		"claude-sonnet-4@20250514",
		"claude-sonnet-4-v1@20250514",
		"claude-sonnet-4-5@20250929",
		"claude-opus-4@20250514",
		"claude-opus-4-v1@20250514",
		"claude-opus-4-1@20250805",
	}

	for _, supportedModel := range supportedModels {
		if model == supportedModel {
			return true
		}
	}
	return false
}

// Message represents a message for Anthropic API
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ToolUse represents a tool call for Anthropic API
type ToolUse struct {
	RecipientName string                 `json:"recipient_name"`
	Name          string                 `json:"name"`
	ID            string                 `json:"id"`
	Input         map[string]interface{} `json:"input"`
	Parameters    map[string]interface{} `json:"parameters"`
}

// ToolResult represents a tool result for Anthropic API
type ToolResult struct {
	Type     string `json:"type"`
	Content  string `json:"content"`
	ToolName string `json:"tool_name"`
}

// CompletionRequest represents a request for Anthropic API
type CompletionRequest struct {
	Model            string         `json:"model,omitempty"`
	Messages         []Message      `json:"messages"`
	MaxTokens        int            `json:"max_tokens,omitempty"`
	Temperature      float64        `json:"temperature,omitempty"`
	TopP             float64        `json:"top_p,omitempty"`
	TopK             int            `json:"top_k,omitempty"`
	StopSequences    []string       `json:"stop_sequences,omitempty"`
	System           string         `json:"system,omitempty"`
	Tools            []Tool         `json:"tools,omitempty"`
	ToolChoice       interface{}    `json:"tool_choice,omitempty"`
	Stream           bool           `json:"stream,omitempty"`
	MetadataKey      string         `json:"metadata,omitempty"`
	AnthropicVersion string         `json:"anthropic_version,omitempty"` // For Vertex AI
	Thinking         *ReasoningSpec `json:"thinking,omitempty"`          // Keep "thinking" for API compatibility
}

// ReasoningSpec represents the reasoning configuration for Anthropic API
// Note: API still uses "thinking" parameter name for compatibility
type ReasoningSpec struct {
	Type         string `json:"type"`                    // "enabled" to enable reasoning
	BudgetTokens int    `json:"budget_tokens,omitempty"` // Optional token budget for reasoning
}

// Tool represents a tool definition for Anthropic API
type Tool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"input_schema"`
}

// ContentBlock represents a content block in Anthropic API response
type ContentBlock struct {
	Type    string   `json:"type"`
	Text    string   `json:"text,omitempty"`
	ToolUse *ToolUse `json:"tool_use,omitempty"`
	// Vertex AI direct fields for tool_use blocks
	ID    string                 `json:"id,omitempty"`
	Name  string                 `json:"name,omitempty"`
	Input map[string]interface{} `json:"input,omitempty"`
}

// CompletionResponse represents a response from Anthropic API
type CompletionResponse struct {
	ID         string         `json:"id"`
	Type       string         `json:"type"`
	Role       string         `json:"role"`
	Content    []ContentBlock `json:"content"`
	Model      string         `json:"model"`
	StopReason string         `json:"stop_reason"`
	Usage      Usage          `json:"usage"`
}

// Usage represents token usage information
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// WithReasoning creates a GenerateOption to set the reasoning mode
// Note: Reasoning parameter is not supported in the current Anthropic API version.
// This option is kept for compatibility but will have no effect.
func WithReasoning(reasoning string) interfaces.GenerateOption {
	return func(options *interfaces.GenerateOptions) {
		if options.LLMConfig == nil {
			options.LLMConfig = &interfaces.LLMConfig{}
		}
		options.LLMConfig.Reasoning = reasoning
		// No actual functionality since reasoning is not supported
	}
}

// Generate generates text from a prompt
func (c *AnthropicClient) Generate(ctx context.Context, prompt string, options ...interfaces.GenerateOption) (string, error) {
	// Check if model is specified
	if c.Model == "" {
		return "", fmt.Errorf("model not specified: use WithModel option when creating the client")
	}

	// Apply options
	params := &interfaces.GenerateOptions{
		LLMConfig: &interfaces.LLMConfig{
			Temperature: 0.7, // Default temperature
		},
	}

	// Apply user-provided options
	for _, option := range options {
		option(params)
	}

	// Check for organization ID in context, and add a default one if missing
	defaultOrgID := "default"
	if id, err := multitenancy.GetOrgID(ctx); err == nil {
		// Organization ID found in context, use it
		ctx = multitenancy.WithOrgID(ctx, id) // Ensure consistency in context
	} else {
		// Add default organization ID to context to prevent errors in tool execution
		ctx = multitenancy.WithOrgID(ctx, defaultOrgID)
	}

	// Build messages with memory and current prompt
	messages := c.buildMessagesWithMemory(ctx, prompt, params)

	// Handle structured output if requested
	if params.ResponseFormat != nil {
		// Convert the schema to a string representation for the prompt
		schemaJSON, err := json.MarshalIndent(params.ResponseFormat.Schema, "", "  ")
		if err != nil {
			return "", fmt.Errorf("failed to marshal response format schema: %w", err)
		}

		// Create an example JSON structure based on the schema
		exampleJSON := createExampleFromSchema(params.ResponseFormat.Schema)
		exampleStr, _ := json.MarshalIndent(exampleJSON, "", "  ")

		// Enhance the user prompt with schema information and example
		// Using best practices from Claude documentation for consistency
		messages[0].Content = fmt.Sprintf(`%s

You must respond with a valid JSON object that exactly follows this schema:
%s

Here is an example of the expected JSON structure:
%s

CRITICAL INSTRUCTIONS:
- Output ONLY valid JSON, no additional text before or after
- Follow the EXACT structure shown in the schema and example
- Use the field names exactly as specified
- Ensure all required fields are present
- Pay special attention to array fields - they must be arrays of objects, not simple objects
- If a field is defined as an array in the schema, it MUST be an array in your response
- The JSON must be directly parsable and match the schema precisely`, prompt, string(schemaJSON), string(exampleStr))

		// Add assistant message prefill to enforce JSON output
		// This helps Claude start the response correctly as JSON
		messages = append(messages, Message{
			Role:    "assistant",
			Content: "{",
		})

		c.logger.Debug(ctx, "Using structured output format with prefill", map[string]interface{}{
			"schema_name": params.ResponseFormat.Name,
		})
	}

	// Create request
	req := CompletionRequest{
		Model:       c.Model,
		Messages:    messages,
		MaxTokens:   2048,
		Temperature: params.LLMConfig.Temperature,
		TopP:        params.LLMConfig.TopP,
	}

	// Add system message if available
	if params.SystemMessage != "" {
		// If structured output is requested, enhance the system message
		if params.ResponseFormat != nil {
			req.System = params.SystemMessage + "\n\nYou must respond with valid JSON that matches the specified schema."
		} else {
			req.System = params.SystemMessage
		}
		c.logger.Debug(ctx, "Using system message", map[string]interface{}{"system_message": req.System})
	} else if params.ResponseFormat != nil {
		// If no system message but structured output is requested, add a system message for JSON
		req.System = "You must respond with valid JSON that matches the specified schema."
		c.logger.Debug(ctx, "Added system message for structured output", nil)
	}

	// Add reasoning parameter if available
	if params.LLMConfig != nil && params.LLMConfig.Reasoning != "" {
		c.logger.Debug(ctx, "Reasoning mode not supported in current API version", map[string]interface{}{"reasoning": params.LLMConfig.Reasoning})
	}

	if params.LLMConfig != nil {
		if len(params.LLMConfig.StopSequences) > 0 {
			req.StopSequences = params.LLMConfig.StopSequences
		}
	}

	var resp CompletionResponse
	var err error

	operation := func() error {
		var apiType string
		if c.VertexConfig != nil && c.VertexConfig.Enabled {
			apiType = "Vertex AI"
		} else {
			apiType = "Anthropic API"
		}

		c.logger.Debug(ctx, "Executing "+apiType+" request", map[string]interface{}{
			"model":          c.Model,
			"temperature":    req.Temperature,
			"top_p":          req.TopP,
			"stop_sequences": req.StopSequences,
			"system":         req.System != "",
		})

		var httpReq *http.Request

		if c.VertexConfig != nil && c.VertexConfig.Enabled {
			// Vertex AI mode
			c.logger.Debug(ctx, "Using Vertex AI endpoint", map[string]interface{}{
				"region":    c.VertexConfig.Region,
				"projectID": c.VertexConfig.ProjectID,
			})

			httpReq, err = c.VertexConfig.CreateVertexHTTPRequest(ctx, &req, "POST", "/v1/messages")
			if err != nil {
				return fmt.Errorf("failed to create Vertex AI request: %w", err)
			}
		} else {
			// Standard Anthropic API mode
			c.logger.Debug(ctx, "Using standard Anthropic API", map[string]interface{}{
				"baseURL": c.BaseURL,
			})

			// Convert request to JSON
			reqBody, err := json.Marshal(req)
			if err != nil {
				return fmt.Errorf("failed to marshal request: %w", err)
			}

			// Create HTTP request
			httpReq, err = http.NewRequestWithContext(
				ctx,
				"POST",
				c.BaseURL+"/v1/messages",
				bytes.NewBuffer(reqBody),
			)
			if err != nil {
				return fmt.Errorf("failed to create request: %w", err)
			}

			// Set headers for standard API
			httpReq.Header.Set("Content-Type", "application/json")
			httpReq.Header.Set("X-API-Key", c.APIKey)
			httpReq.Header.Set("Anthropic-Version", "2023-06-01")
		}

		// Send request
		httpResp, err := c.HTTPClient.Do(httpReq)
		if err != nil {
			c.logger.Error(ctx, "Error from Anthropic API", map[string]interface{}{
				"error": err.Error(),
				"model": c.Model,
			})
			return fmt.Errorf("failed to send request: %w", err)
		}
		defer func() {
			if closeErr := httpResp.Body.Close(); closeErr != nil {
				c.logger.Warn(ctx, "Failed to close response body", map[string]interface{}{
					"error": closeErr.Error(),
				})
			}
		}()

		// Read response body
		respBody, err := io.ReadAll(httpResp.Body)
		if err != nil {
			return fmt.Errorf("failed to read response body: %w", err)
		}

		// Check for error response
		if httpResp.StatusCode != http.StatusOK {
			c.logger.Error(ctx, "Error from Anthropic API", map[string]interface{}{
				"status_code": httpResp.StatusCode,
				"response":    string(respBody),
				"model":       c.Model,
			})
			return fmt.Errorf("error from Anthropic API: %s", string(respBody))
		}

		// Unmarshal response
		err = json.Unmarshal(respBody, &resp)
		if err != nil {
			return fmt.Errorf("failed to unmarshal response: %w", err)
		}

		return nil
	}

	if c.vertexRetryExecutor != nil {
		c.logger.Info(ctx, "Using Vertex retry mechanism with region rotation", map[string]interface{}{
			"model":          c.Model,
			"current_region": c.VertexConfig.GetCurrentRegion(),
		})
		err = c.vertexRetryExecutor.Execute(ctx, operation)
	} else if c.retryExecutor != nil {
		c.logger.Info(ctx, "Using standard retry mechanism for Anthropic request", map[string]interface{}{
			"model":                   c.Model,
			"vertex_config_available": c.VertexConfig != nil,
		})
		err = c.retryExecutor.Execute(ctx, operation)
	} else {
		c.logger.Debug(ctx, "No retry mechanism configured", map[string]interface{}{
			"model": c.Model,
		})
		err = operation()
	}

	if err != nil {
		return "", err
	}

	// Extract text from content blocks
	var contentText []string
	for _, block := range resp.Content {
		if block.Type == "text" {
			contentText = append(contentText, block.Text)
		}
	}

	if len(contentText) == 0 {
		return "", fmt.Errorf("no text content in response")
	}

	response := strings.Join(contentText, "\n")

	// For structured output, prepend the opening brace that was used as prefill
	if params.ResponseFormat != nil && !strings.HasPrefix(strings.TrimSpace(response), "{") {
		response = "{" + response
	}

	c.logger.Debug(ctx, "Successfully received response from Anthropic", map[string]interface{}{
		"model":             c.Model,
		"structured_output": params.ResponseFormat != nil,
		"response_length":   len(response),
		"response_preview": func() string {

			return response
		}(),
	})

	return response, nil
}

// Chat uses the messages API to have a conversation with a model
func (c *AnthropicClient) Chat(ctx context.Context, messages []llm.Message, params *llm.GenerateParams) (string, error) {
	// Check if model is specified
	if c.Model == "" {
		return "", fmt.Errorf("model not specified: use WithModel option when creating the client")
	}

	if params == nil {
		params = llm.DefaultGenerateParams()
	}

	// Convert messages to the Anthropic Chat format
	anthropicMessages := make([]Message, len(messages))
	var systemMessage string

	for i, msg := range messages {
		// Check if it's a system message
		if msg.Role == "system" {
			systemMessage = msg.Content
			// Skip this message in the regular messages array
			continue
		}

		// Map role names (Anthropic uses "assistant" and "user")
		role := msg.Role
		switch role {
		case "assistant", "user":
			// These roles are the same in Anthropic
		case "tool":
			// Tool messages need special handling
			// For simplicity, we'll convert them to assistant messages
			role = "assistant"
		}

		anthropicMessages[i] = Message{
			Role:    role,
			Content: msg.Content,
		}
	}

	// Filter out any nil messages (from system messages being skipped) and messages with empty content
	var filteredMessages []Message
	for _, msg := range anthropicMessages {
		if msg.Role != "" && strings.TrimSpace(msg.Content) != "" {
			filteredMessages = append(filteredMessages, msg)
		}
	}

	// Create chat request
	req := CompletionRequest{
		Model:         c.Model,
		Messages:      filteredMessages,
		MaxTokens:     2048,
		Temperature:   params.Temperature,
		TopP:          params.TopP,
		StopSequences: params.StopSequences,
	}

	// Add system message if available
	if systemMessage != "" {
		req.System = systemMessage
	}

	// Add reasoning parameter if available
	if params.Reasoning != "" {
		c.logger.Debug(ctx, "Reasoning mode not supported in current API version", map[string]interface{}{"reasoning": params.Reasoning})
	}

	var resp CompletionResponse
	var err error

	operation := func() error {
		var apiType string
		if c.VertexConfig != nil && c.VertexConfig.Enabled {
			apiType = "Vertex AI"
		} else {
			apiType = "Anthropic API"
		}

		c.logger.Debug(ctx, "Executing "+apiType+" Chat request", map[string]interface{}{
			"model":          c.Model,
			"temperature":    req.Temperature,
			"top_p":          req.TopP,
			"stop_sequences": req.StopSequences,
			"messages":       len(req.Messages),
		})

		var httpReq *http.Request

		if c.VertexConfig != nil && c.VertexConfig.Enabled {
			// Vertex AI mode
			httpReq, err = c.VertexConfig.CreateVertexHTTPRequest(ctx, &req, "POST", "/v1/messages")
			if err != nil {
				return fmt.Errorf("failed to create Vertex AI chat request: %w", err)
			}
		} else {
			// Standard Anthropic API mode
			// Convert request to JSON
			reqBody, err := json.Marshal(req)
			if err != nil {
				return fmt.Errorf("failed to marshal request: %w", err)
			}

			// Create HTTP request
			httpReq, err = http.NewRequestWithContext(
				ctx,
				"POST",
				c.BaseURL+"/v1/messages",
				bytes.NewBuffer(reqBody),
			)
			if err != nil {
				return fmt.Errorf("failed to create request: %w", err)
			}

			// Set headers for standard API
			httpReq.Header.Set("Content-Type", "application/json")
			httpReq.Header.Set("X-API-Key", c.APIKey)
			httpReq.Header.Set("Anthropic-Version", "2023-06-01")
		}

		// Send request
		httpResp, err := c.HTTPClient.Do(httpReq)
		if err != nil {
			c.logger.Error(ctx, "Error from Anthropic Chat API", map[string]interface{}{
				"error": err.Error(),
				"model": c.Model,
			})
			return fmt.Errorf("failed to send request: %w", err)
		}
		defer func() {
			if closeErr := httpResp.Body.Close(); closeErr != nil {
				c.logger.Warn(ctx, "Failed to close response body", map[string]interface{}{
					"error": closeErr.Error(),
				})
			}
		}()

		// Read response body
		respBody, err := io.ReadAll(httpResp.Body)
		if err != nil {
			return fmt.Errorf("failed to read response body: %w", err)
		}

		// Check for error response
		if httpResp.StatusCode != http.StatusOK {
			c.logger.Error(ctx, "Error from Anthropic Chat API", map[string]interface{}{
				"status_code": httpResp.StatusCode,
				"response":    string(respBody),
				"model":       c.Model,
			})
			return fmt.Errorf("error from Anthropic API: %s", string(respBody))
		}

		// Log raw response before unmarshaling for debugging
		c.logger.Debug(ctx, "Raw streaming response before unmarshaling", map[string]interface{}{
			"response_length": len(respBody),
			"response_prefix": func() string {
				if len(respBody) > 100 {
					return string(respBody[:100])
				}
				return string(respBody)
			}(),
			"first_char": func() string {
				if len(respBody) > 0 {
					return fmt.Sprintf("'%c' (0x%02x)", respBody[0], respBody[0])
				}
				return "empty"
			}(),
		})

		// Unmarshal response
		err = json.Unmarshal(respBody, &resp)
		if err != nil {
			c.logger.Error(ctx, "Failed to unmarshal streaming response", map[string]interface{}{
				"error":           err.Error(),
				"response_length": len(respBody),
				"response_sample": func() string {
					if len(respBody) > 200 {
						return string(respBody[:200])
					}
					return string(respBody)
				}(),
			})
			return fmt.Errorf("failed to unmarshal response: %w", err)
		}

		return nil
	}

	if c.vertexRetryExecutor != nil {
		c.logger.Info(ctx, "Using Vertex retry mechanism with region rotation for Chat", map[string]interface{}{
			"model":          c.Model,
			"current_region": c.VertexConfig.GetCurrentRegion(),
		})
		err = c.vertexRetryExecutor.Execute(ctx, operation)
	} else if c.retryExecutor != nil {
		c.logger.Info(ctx, "Using standard retry mechanism for Anthropic Chat request", map[string]interface{}{
			"model":                   c.Model,
			"vertex_config_available": c.VertexConfig != nil,
		})
		err = c.retryExecutor.Execute(ctx, operation)
	} else {
		c.logger.Debug(ctx, "No retry mechanism configured for Chat request", map[string]interface{}{
			"model": c.Model,
		})
		err = operation()
	}

	if err != nil {
		return "", err
	}

	// Extract text from content blocks
	var contentText []string
	for _, block := range resp.Content {
		if block.Type == "text" {
			contentText = append(contentText, block.Text)
		}
	}

	if len(contentText) == 0 {
		return "", fmt.Errorf("no text content in response")
	}

	response := strings.Join(contentText, "\n")

	c.logger.Debug(ctx, "Successfully received chat response from Anthropic", map[string]interface{}{
		"model":           c.Model,
		"response_length": len(response),
		"response_preview": func() string {

			return response
		}(),
	})

	return response, nil
}

// GenerateWithTools implements interfaces.LLM.GenerateWithTools
func (c *AnthropicClient) GenerateWithTools(ctx context.Context, prompt string, tools []interfaces.Tool, options ...interfaces.GenerateOption) (string, error) {
	// Check if model is specified
	if c.Model == "" {
		return "", fmt.Errorf("model not specified: use WithModel option when creating the client")
	}

	// Apply options
	params := &interfaces.GenerateOptions{
		LLMConfig: &interfaces.LLMConfig{
			Temperature: 0.7, // Default temperature
		},
	}

	// Apply user-provided options
	for _, opt := range options {
		if opt != nil {
			opt(params)
		}
	}

	// Set default max iterations if not provided
	maxIterations := params.MaxIterations
	if maxIterations == 0 {
		maxIterations = 2 // Default to current behavior
	}

	// Check for organization ID in context, and add a default one if missing
	defaultOrgID := "default"
	if id, err := multitenancy.GetOrgID(ctx); err == nil {
		// Organization ID found in context, use it
		ctx = multitenancy.WithOrgID(ctx, id) // Ensure consistency in context
	} else {
		// Add default organization ID to context to prevent errors in tool execution
		ctx = multitenancy.WithOrgID(ctx, defaultOrgID)
	}

	// Convert tools to Anthropic format
	anthropicTools := make([]Tool, len(tools))
	for i, tool := range tools {
		// Convert ParameterSpec to JSON Schema
		properties := make(map[string]interface{})
		required := []string{}

		for name, param := range tool.Parameters() {
			properties[name] = map[string]interface{}{
				"type":        param.Type,
				"description": param.Description,
			}
			if param.Default != nil {
				properties[name].(map[string]interface{})["default"] = param.Default
			}
			if param.Required {
				required = append(required, name)
			}
			if param.Items != nil {
				properties[name].(map[string]interface{})["items"] = map[string]interface{}{
					"type": param.Items.Type,
				}
				if param.Items.Enum != nil {
					properties[name].(map[string]interface{})["items"].(map[string]interface{})["enum"] = param.Items.Enum
				}
			}
			if param.Enum != nil {
				properties[name].(map[string]interface{})["enum"] = param.Enum
			}
		}

		// Create the input schema for this tool
		inputSchema := map[string]interface{}{
			"type":       "object",
			"properties": properties,
			"required":   required,
		}

		anthropicTools[i] = Tool{
			Name:        tool.Name(),
			Description: tool.Description(),
			InputSchema: inputSchema,
		}
	}

	// Track tool call repetitions for loop detection
	toolCallHistory := make(map[string]int)

	// Build messages with memory and current prompt
	messages := c.buildMessagesWithMemory(ctx, prompt, params)

	// Iterative tool calling loop
	for iteration := 0; iteration < maxIterations; iteration++ {
		// Create request
		req := CompletionRequest{
			Model:       c.Model,
			Messages:    messages,
			MaxTokens:   2048,
			Temperature: params.LLMConfig.Temperature,
			TopP:        params.LLMConfig.TopP,
			Tools:       anthropicTools,
			// Auto use tools when needed
			ToolChoice: map[string]string{
				"type": "auto",
			},
		}

		// Add system message if available
		if params.SystemMessage != "" {
			// If structured output is requested, enhance the system message to ensure raw JSON
			if params.ResponseFormat != nil {
				req.System = params.SystemMessage + "\n\nIMPORTANT: You must respond with valid JSON that matches the specified schema. Return ONLY the raw JSON object without any markdown formatting, code blocks, or wrapper text. Pay special attention to array fields - if a field is defined as an array in the schema, it MUST be an array in your response, not an object."
			} else {
				req.System = params.SystemMessage
			}
			c.logger.Debug(ctx, "Using system message", map[string]interface{}{"system_message": params.SystemMessage})
		} else if params.ResponseFormat != nil {
			// If no system message but structured output is requested, add a system message for JSON
			req.System = "You must respond with valid JSON that matches the specified schema. Return ONLY the raw JSON object without any markdown formatting, code blocks, or wrapper text. Pay special attention to array fields - if a field is defined as an array in the schema, it MUST be an array in your response, not an object."
			c.logger.Debug(ctx, "Added system message for structured output", nil)
		}

		// Add reasoning parameter if available
		if params.LLMConfig != nil && params.LLMConfig.Reasoning != "" {
			c.logger.Debug(ctx, "Reasoning mode not supported in current API version", map[string]interface{}{"reasoning": params.LLMConfig.Reasoning})
		}

		// Send request
		c.logger.Debug(ctx, "Sending request with tools to Anthropic", map[string]interface{}{
			"model":         c.Model,
			"temperature":   req.Temperature,
			"top_p":         req.TopP,
			"messages":      len(req.Messages),
			"tools":         len(req.Tools),
			"system":        req.System != "",
			"iteration":     iteration + 1,
			"maxIterations": maxIterations,
		})

		var resp CompletionResponse
		var err error

		// Define operation for retry mechanism
		operation := func() error {
			// Create HTTP request (supports both Vertex AI and standard Anthropic API)
			httpReq, err := c.createHTTPRequest(ctx, &req, "/v1/messages")
			if err != nil {
				return fmt.Errorf("failed to create request (iteration %d): %w", iteration+1, err)
			}

			// Send request
			httpResp, err := c.HTTPClient.Do(httpReq)
			if err != nil {
				c.logger.Error(ctx, "Error from Anthropic API", map[string]interface{}{
					"error":     err.Error(),
					"model":     c.Model,
					"iteration": iteration + 1,
				})
				return fmt.Errorf("failed to send request (iteration %d): %w", iteration+1, err)
			}
			defer func() {
				if closeErr := httpResp.Body.Close(); closeErr != nil {
					c.logger.Warn(ctx, "Failed to close response body", map[string]interface{}{
						"error": closeErr.Error(),
					})
				}
			}()

			// Read response body
			respBody, err := io.ReadAll(httpResp.Body)
			if err != nil {
				return fmt.Errorf("failed to read response body (iteration %d): %w", iteration+1, err)
			}

			// Check for error response
			if httpResp.StatusCode != http.StatusOK {
				c.logger.Error(ctx, "Error from Anthropic API", map[string]interface{}{
					"status_code": httpResp.StatusCode,
					"response":    string(respBody),
					"model":       c.Model,
					"iteration":   iteration + 1,
				})
				return fmt.Errorf("error from Anthropic API (iteration %d): %s", iteration+1, string(respBody))
			}

			// Log raw response before unmarshaling for debugging
			c.logger.Debug(ctx, "Raw response before unmarshaling", map[string]interface{}{
				"response_length": len(respBody),
				"response_prefix": func() string {
					if len(respBody) > 100 {
						return string(respBody[:100])
					}
					return string(respBody)
				}(),
				"first_char": func() string {
					if len(respBody) > 0 {
						return fmt.Sprintf("'%c' (0x%02x)", respBody[0], respBody[0])
					}
					return "empty"
				}(),
				"iteration": iteration + 1,
			})

			// Unmarshal response
			err = json.Unmarshal(respBody, &resp)
			if err != nil {
				return fmt.Errorf("failed to unmarshal response (iteration %d): %w", iteration+1, err)
			}

			return nil
		}

		// Execute operation with retry mechanism
		if c.vertexRetryExecutor != nil {
			c.logger.Info(ctx, "Using Vertex retry mechanism with region rotation for GenerateWithTools", map[string]interface{}{
				"model":          c.Model,
				"current_region": c.VertexConfig.GetCurrentRegion(),
				"iteration":      iteration + 1,
			})
			err = c.vertexRetryExecutor.Execute(ctx, operation)
		} else if c.retryExecutor != nil {
			c.logger.Info(ctx, "Using standard retry mechanism for GenerateWithTools", map[string]interface{}{
				"model":                   c.Model,
				"vertex_config_available": c.VertexConfig != nil,
				"iteration":               iteration + 1,
			})
			err = c.retryExecutor.Execute(ctx, operation)
		} else {
			c.logger.Debug(ctx, "No retry mechanism configured for GenerateWithTools", map[string]interface{}{
				"model":     c.Model,
				"iteration": iteration + 1,
			})
			err = operation()
		}

		if err != nil {
			return "", err
		}

		// Make sure content is not nil
		if resp.Content == nil {
			c.logger.Error(ctx, "No content in response", map[string]interface{}{"iteration": iteration + 1})
			return "", fmt.Errorf("no content in response (iteration %d)", iteration+1)
		}

		// Check if the model wants to use tools
		var hasToolUse bool
		var toolCalls []ToolUse
		var textContent []string

		c.logger.Debug(ctx, "Response content blocks", map[string]interface{}{
			"numBlocks": len(resp.Content),
			"iteration": iteration + 1,
			"blockTypes": func() []string {
				types := make([]string, len(resp.Content))
				for i, block := range resp.Content {
					types[i] = block.Type
					if block.Type == "tool_use" && block.ToolUse != nil {
						toolName := ""
						if block.ToolUse.Name != "" {
							toolName = block.ToolUse.Name
						} else if block.ToolUse.RecipientName != "" {
							toolName = block.ToolUse.RecipientName
						}
						c.logger.Debug(ctx, "Found tool use block", map[string]interface{}{
							"toolName":  toolName,
							"toolID":    block.ToolUse.ID,
							"iteration": iteration + 1,
						})
					}
				}
				return types
			}(),
		})

		for _, contentBlock := range resp.Content {
			switch contentBlock.Type {
			case "tool_use":
				hasToolUse = true
				// Handle both nested ToolUse (direct API) and direct fields (Vertex AI)
				if contentBlock.ToolUse != nil {
					toolCalls = append(toolCalls, *contentBlock.ToolUse)
				} else if contentBlock.ID != "" && contentBlock.Name != "" {
					// Create ToolUse from direct fields (Vertex AI format)
					toolUse := ToolUse{
						ID:    contentBlock.ID,
						Name:  contentBlock.Name,
						Input: contentBlock.Input,
					}
					toolCalls = append(toolCalls, toolUse)
				}
			case "text":
				textContent = append(textContent, contentBlock.Text)
			}
		}

		c.logger.Debug(ctx, "Tool use detection results", map[string]interface{}{
			"hasToolUse": hasToolUse,
			"toolCalls":  len(toolCalls),
			"iteration":  iteration + 1,
		})

		// If no tool use, return the text content
		if !hasToolUse {
			if len(textContent) == 0 {
				return "", fmt.Errorf("no text content in response (iteration %d)", iteration+1)
			}

			// Join the text content
			response := strings.Join(textContent, "\n")

			// If we have a ResponseFormat, extract JSON from the response
			if params.ResponseFormat != nil {
				extractedJSON := extractJSONFromResponse(response)
				if extractedJSON != response {
					c.logger.Debug(ctx, "Extracted JSON from response", map[string]interface{}{
						"original_length":  len(response),
						"extracted_length": len(extractedJSON),
					})
					response = extractedJSON
				}
			}

			c.logger.Debug(ctx, "Returning final response (no tool use)", map[string]interface{}{
				"response_length": len(response),
				"response_preview": func() string {

					return response
				}(),
				"iteration": iteration + 1,
			})

			return response, nil
		}

		// The model wants to use tools
		c.logger.Info(ctx, "Processing tool calls", map[string]interface{}{
			"count":     len(toolCalls),
			"iteration": iteration + 1,
		})

		// Add the assistant response to messages only if there's text content
		// (Tool-only responses will have empty text content)
		assistantContent := strings.Join(textContent, "\n")
		if strings.TrimSpace(assistantContent) != "" {
			messages = append(messages, Message{
				Role:    "assistant",
				Content: assistantContent,
			})
		}

		// Process each tool call
		var toolResults []ToolResult
		for _, toolCall := range toolCalls {
			// Get tool name - it could be in either Name or RecipientName field
			toolName := ""
			if toolCall.Name != "" {
				toolName = toolCall.Name
			} else if toolCall.RecipientName != "" {
				toolName = toolCall.RecipientName
			} else {
				c.logger.Error(ctx, "Tool call missing both Name and RecipientName", map[string]interface{}{"iteration": iteration + 1})
				continue
			}

			// Find the requested tool
			var selectedTool interfaces.Tool
			for _, tool := range tools {
				if tool.Name() == toolName {
					selectedTool = tool
					break
				}
			}

			if selectedTool == nil {
				c.logger.Error(ctx, "Tool not found", map[string]interface{}{
					"toolName":  toolName,
					"iteration": iteration + 1,
					"availableTools": func() []string {
						names := make([]string, len(tools))
						for i, t := range tools {
							names[i] = t.Name()
						}
						return names
					}(),
				})

				// Add tool not found error as tool result instead of returning
				errorMessage := fmt.Sprintf("Error: tool not found: %s", toolName)

				// Store failed tool call in memory if provided
				if params.Memory != nil {
					_ = params.Memory.AddMessage(ctx, interfaces.Message{
						Role:    "assistant",
						Content: "",
						ToolCalls: []interfaces.ToolCall{{
							ID:        toolCall.ID,
							Name:      toolName,
							Arguments: "{}",
						}},
					})
					_ = params.Memory.AddMessage(ctx, interfaces.Message{
						Role:       "tool",
						Content:    errorMessage,
						ToolCallID: toolCall.ID,
						Metadata: map[string]interface{}{
							"tool_name": toolName,
						},
					})
				}

				// Add error as tool result
				toolResults = append(toolResults, ToolResult{
					Type:     "tool_result",
					Content:  errorMessage,
					ToolName: toolName,
				})

				continue // Continue processing other tool calls
			}

			// Get parameters - could be in either Input or Parameters field
			var parameters map[string]interface{}
			if len(toolCall.Input) > 0 {
				parameters = toolCall.Input
			} else if len(toolCall.Parameters) > 0 {
				parameters = toolCall.Parameters
			}

			// Convert parameters to JSON string
			toolCallJSON, err := json.Marshal(parameters)
			if err != nil {
				c.logger.Error(ctx, "Error marshalling parameters", map[string]interface{}{
					"error":      err.Error(),
					"parameters": parameters,
					"iteration":  iteration + 1,
				})
				return "", fmt.Errorf("error marshalling parameters (iteration %d): %w", iteration+1, err)
			}

			// Log parameters for debugging
			c.logger.Debug(ctx, "Tool parameters", map[string]interface{}{
				"toolName":   toolName,
				"parameters": string(toolCallJSON),
				"iteration":  iteration + 1,
			})

			// Execute the tool
			c.logger.Info(ctx, "Executing tool", map[string]interface{}{
				"toolName":  selectedTool.Name(),
				"iteration": iteration + 1,
			})
			toolResult, err := selectedTool.Execute(ctx, string(toolCallJSON))

			// Check for repetitive calls and add warning if needed
			cacheKey := toolName + ":" + string(toolCallJSON)
			toolCallHistory[cacheKey]++

			if toolCallHistory[cacheKey] > 2 {
				warning := fmt.Sprintf("\n\n[WARNING: This is call #%d to %s with identical parameters. You may be in a loop. Consider using the available information to provide a final answer.]",
					toolCallHistory[cacheKey],
					toolName)
				if err == nil {
					toolResult += warning
				}
				c.logger.Warn(ctx, "Repetitive tool call detected", map[string]interface{}{
					"toolName":  toolName,
					"callCount": toolCallHistory[cacheKey],
					"iteration": iteration + 1,
				})
			}

			// Store tool call and result in memory if provided
			if params.Memory != nil {
				if err != nil {
					// Store failed tool call result
					_ = params.Memory.AddMessage(ctx, interfaces.Message{
						Role:    "assistant",
						Content: "",
						ToolCalls: []interfaces.ToolCall{{
							ID:        toolCall.ID,
							Name:      toolName,
							Arguments: string(toolCallJSON),
						}},
					})
					_ = params.Memory.AddMessage(ctx, interfaces.Message{
						Role:       "tool",
						Content:    fmt.Sprintf("Error: %v", err),
						ToolCallID: toolCall.ID,
						Metadata: map[string]interface{}{
							"tool_name": toolName,
						},
					})
				} else {
					// Store successful tool call and result
					_ = params.Memory.AddMessage(ctx, interfaces.Message{
						Role:    "assistant",
						Content: "",
						ToolCalls: []interfaces.ToolCall{{
							ID:        toolCall.ID,
							Name:      toolName,
							Arguments: string(toolCallJSON),
						}},
					})
					_ = params.Memory.AddMessage(ctx, interfaces.Message{
						Role:       "tool",
						Content:    toolResult,
						ToolCallID: toolCall.ID,
						Metadata: map[string]interface{}{
							"tool_name": toolName,
						},
					})
				}
			}

			if err != nil {
				c.logger.Error(ctx, "Error executing tool", map[string]interface{}{
					"toolName":  selectedTool.Name(),
					"error":     err.Error(),
					"iteration": iteration + 1,
				})
				// Return error as tool result
				toolResults = append(toolResults, ToolResult{
					Type:     "tool_result",
					Content:  fmt.Sprintf("Error: %v", err),
					ToolName: toolName,
				})
				continue
			}

			// Add tool result
			toolResults = append(toolResults, ToolResult{
				Type:     "tool_result",
				Content:  toolResult,
				ToolName: toolName,
			})
		}

		// Create a new message from the user with the tool results
		toolResultsJSON, err := json.Marshal(toolResults)
		if err != nil {
			return "", fmt.Errorf("failed to marshal tool results (iteration %d): %w", iteration+1, err)
		}

		// Add a user message with the tool results
		messages = append(messages, Message{
			Role:    "user",
			Content: fmt.Sprintf("Here are the tool results: %s", string(toolResultsJSON)),
		})

		// Continue to the next iteration with updated messages
	}

	// If we've reached the maximum iterations and the model is still requesting tools,
	// make one final call without tools to get a conclusion
	c.logger.Info(ctx, "Maximum iterations reached, making final call without tools", map[string]interface{}{
		"maxIterations": maxIterations,
	})

	// Create a final request without tools to force the LLM to provide a conclusion
	finalReq := CompletionRequest{
		Model:       c.Model,
		Messages:    messages,
		MaxTokens:   2048,
		Temperature: params.LLMConfig.Temperature,
		TopP:        params.LLMConfig.TopP,
		Tools:       nil, // No tools for final call
	}

	// Add system message if available and enhance for structured output
	if params.SystemMessage != "" {
		if params.ResponseFormat != nil {
			finalReq.System = params.SystemMessage + "\n\nIMPORTANT: You must respond with valid JSON that matches the specified schema. Return ONLY the raw JSON object without any markdown formatting, code blocks, or wrapper text. Pay special attention to array fields - if a field is defined as an array in the schema, it MUST be an array in your response, not an object."
		} else {
			finalReq.System = params.SystemMessage
		}
	} else if params.ResponseFormat != nil {
		// If no system message but structured output is requested, add a system message for JSON
		finalReq.System = "You must respond with valid JSON that matches the specified schema. Return ONLY the raw JSON object without any markdown formatting, code blocks, or wrapper text. Pay special attention to array fields - if a field is defined as an array in the schema, it MUST be an array in your response, not an object."
	}

	// Add a user message to encourage conclusion
	finalUserMessage := "Please provide your final response based on the information available. Do not request any additional tools."

	// If structured output is requested, enhance the final message with schema and examples
	if params.ResponseFormat != nil {
		// Convert the schema to a string representation for the prompt
		schemaJSON, err := json.MarshalIndent(params.ResponseFormat.Schema, "", "  ")
		if err == nil {
			// Create an example JSON structure based on the schema
			exampleJSON := createExampleFromSchema(params.ResponseFormat.Schema)
			exampleStr, _ := json.MarshalIndent(exampleJSON, "", "  ")

			// Enhance the final user message with schema information and example
			finalUserMessage = fmt.Sprintf(`%s

You must respond with a valid JSON object that exactly follows this schema:
%s

Here is an example of the expected JSON structure:
%s

CRITICAL INSTRUCTIONS:
- Output ONLY valid JSON, no additional text before or after
- Follow the EXACT structure shown in the schema and example
- Use the field names exactly as specified
- Ensure all required fields are present
- Pay special attention to array fields - they must be arrays of objects, not simple objects
- If a field is defined as an array in the schema, it MUST be an array in your response
- The JSON must be directly parsable and match the schema precisely`, finalUserMessage, string(schemaJSON), string(exampleStr))
		}
	}

	messages = append(messages, Message{
		Role:    "user",
		Content: finalUserMessage,
	})

	// Add assistant message prefill for structured output to enforce JSON output
	if params.ResponseFormat != nil {
		messages = append(messages, Message{
			Role:    "assistant",
			Content: "{",
		})
		c.logger.Debug(ctx, "Added prefill for structured output in final call", nil)
	}

	finalReq.Messages = messages

	c.logger.Debug(ctx, "Making final request without tools", map[string]interface{}{
		"messages": len(finalReq.Messages),
	})

	// Create final HTTP request (supports both Vertex AI and standard Anthropic API)
	finalHTTPReq, err := c.createHTTPRequest(ctx, &finalReq, "/v1/messages")
	if err != nil {
		return "", fmt.Errorf("failed to create final request: %w", err)
	}

	// Send final request
	finalHTTPResp, err := c.HTTPClient.Do(finalHTTPReq)
	if err != nil {
		c.logger.Error(ctx, "Error in final call without tools", map[string]interface{}{"error": err.Error()})
		return "", fmt.Errorf("failed to send final request: %w", err)
	}
	defer func() {
		if closeErr := finalHTTPResp.Body.Close(); closeErr != nil {
			c.logger.Warn(ctx, "Failed to close final response body", map[string]interface{}{
				"error": closeErr.Error(),
			})
		}
	}()

	// Read final response body
	finalRespBody, err := io.ReadAll(finalHTTPResp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read final response body: %w", err)
	}

	// Check for error response
	if finalHTTPResp.StatusCode != http.StatusOK {
		c.logger.Error(ctx, "Error from Anthropic API in final call", map[string]interface{}{
			"status_code": finalHTTPResp.StatusCode,
			"response":    string(finalRespBody),
		})
		return "", fmt.Errorf("error from Anthropic API in final call: %s", string(finalRespBody))
	}

	// Log raw final response before unmarshaling for debugging
	c.logger.Debug(ctx, "Raw final response before unmarshaling", map[string]interface{}{
		"response_length": len(finalRespBody),
		"response_prefix": func() string {
			if len(finalRespBody) > 100 {
				return string(finalRespBody[:100])
			}
			return string(finalRespBody)
		}(),
		"first_char": func() string {
			if len(finalRespBody) > 0 {
				return fmt.Sprintf("'%c' (0x%02x)", finalRespBody[0], finalRespBody[0])
			}
			return "empty"
		}(),
	})

	// Unmarshal final response
	var finalResp CompletionResponse
	err = json.Unmarshal(finalRespBody, &finalResp)
	if err != nil {
		c.logger.Error(ctx, "Failed to unmarshal final response", map[string]interface{}{
			"error":           err.Error(),
			"response_length": len(finalRespBody),
			"response_sample": func() string {
				if len(finalRespBody) > 200 {
					return string(finalRespBody[:200])
				}
				return string(finalRespBody)
			}(),
		})
		return "", fmt.Errorf("failed to unmarshal final response: %w", err)
	}

	// Extract text content from final response
	if finalResp.Content == nil {
		return "", fmt.Errorf("no content in final response")
	}

	var finalTextContent []string
	for _, contentBlock := range finalResp.Content {
		if contentBlock.Type == "text" {
			finalTextContent = append(finalTextContent, contentBlock.Text)
		}
	}

	if len(finalTextContent) == 0 {
		return "", fmt.Errorf("no text content in final response")
	}

	response := strings.Join(finalTextContent, "\n")

	// For structured output, prepend the opening brace that was used as prefill
	if params.ResponseFormat != nil {
		response = "{" + response
	}

	// If we have a ResponseFormat, extract JSON from the response
	if params.ResponseFormat != nil {
		extractedJSON := extractJSONFromResponse(response)
		if extractedJSON != response {
			c.logger.Debug(ctx, "Extracted JSON from final response", map[string]interface{}{
				"original_length":  len(response),
				"extracted_length": len(extractedJSON),
			})
			response = extractedJSON
		}
	}

	c.logger.Info(ctx, "Successfully received final response without tools", map[string]interface{}{
		"response_length": len(response),
		"response_preview": func() string {

			return response
		}(),
	})

	return response, nil
}

// createHTTPRequest creates an HTTP request for either Vertex AI or standard Anthropic API
func (c *AnthropicClient) createHTTPRequest(ctx context.Context, req *CompletionRequest, path string) (*http.Request, error) {
	if c.VertexConfig != nil && c.VertexConfig.Enabled {
		// Vertex AI mode
		return c.VertexConfig.CreateVertexHTTPRequest(ctx, req, "POST", path)
	} else {
		// Standard Anthropic API mode
		// Convert request to JSON
		reqBody, err := json.Marshal(req)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request: %w", err)
		}

		// Create HTTP request
		httpReq, err := http.NewRequestWithContext(
			ctx,
			"POST",
			c.BaseURL+path,
			bytes.NewBuffer(reqBody),
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		// Set headers for standard API
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("X-API-Key", c.APIKey)
		httpReq.Header.Set("Anthropic-Version", "2023-06-01")

		return httpReq, nil
	}
}

// createStreamingHTTPRequest creates an HTTP request for streaming, supporting both Vertex AI and standard API
func (c *AnthropicClient) createStreamingHTTPRequest(ctx context.Context, req *CompletionRequest, path string) (*http.Request, error) {
	if c.VertexConfig != nil && c.VertexConfig.Enabled {
		// Vertex AI mode
		return c.VertexConfig.CreateVertexStreamingHTTPRequest(ctx, req, "POST", path)
	} else {
		// Standard Anthropic API mode
		// Ensure streaming is enabled
		req.Stream = true

		// Convert request to JSON
		reqBody, err := json.Marshal(req)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request: %w", err)
		}

		// Create HTTP request
		httpReq, err := http.NewRequestWithContext(
			ctx,
			"POST",
			c.BaseURL+path,
			bytes.NewBuffer(reqBody),
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		// Set headers for standard API with streaming
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("X-API-Key", c.APIKey)
		httpReq.Header.Set("Anthropic-Version", "2023-06-01")
		httpReq.Header.Set("Accept", "text/event-stream")
		httpReq.Header.Set("Cache-Control", "no-cache")

		return httpReq, nil
	}
}

// Name implements interfaces.LLM.Name
func (c *AnthropicClient) Name() string {
	return "anthropic"
}

// SupportsStreaming implements interfaces.LLM.SupportsStreaming
func (c *AnthropicClient) SupportsStreaming() bool {
	return true
}

// WithTemperature creates a GenerateOption to set the temperature
func WithTemperature(temperature float64) interfaces.GenerateOption {
	return func(options *interfaces.GenerateOptions) {
		options.LLMConfig.Temperature = temperature
	}
}

// WithTopP creates a GenerateOption to set the top_p
func WithTopP(topP float64) interfaces.GenerateOption {
	return func(options *interfaces.GenerateOptions) {
		options.LLMConfig.TopP = topP
	}
}

// WithFrequencyPenalty creates a GenerateOption to set the frequency penalty
func WithFrequencyPenalty(frequencyPenalty float64) interfaces.GenerateOption {
	return func(options *interfaces.GenerateOptions) {
		options.LLMConfig.FrequencyPenalty = frequencyPenalty
	}
}

// WithPresencePenalty creates a GenerateOption to set the presence penalty
func WithPresencePenalty(presencePenalty float64) interfaces.GenerateOption {
	return func(options *interfaces.GenerateOptions) {
		options.LLMConfig.PresencePenalty = presencePenalty
	}
}

// WithStopSequences creates a GenerateOption to set the stop sequences
func WithStopSequences(stopSequences []string) interfaces.GenerateOption {
	return func(options *interfaces.GenerateOptions) {
		options.LLMConfig.StopSequences = stopSequences
	}
}

// WithSystemMessage creates a GenerateOption to set the system message
func WithSystemMessage(systemMessage string) interfaces.GenerateOption {
	return func(options *interfaces.GenerateOptions) {
		options.SystemMessage = systemMessage
	}
}

// WithResponseFormat creates a GenerateOption to set the response format
func WithResponseFormat(format interfaces.ResponseFormat) interfaces.GenerateOption {
	return func(options *interfaces.GenerateOptions) {
		options.ResponseFormat = &format
	}
}

// createExampleFromSchema creates an example JSON structure based on the schema
func createExampleFromSchema(schema map[string]interface{}) map[string]interface{} {
	example := make(map[string]interface{})

	// Check if schema has properties
	if properties, ok := schema["properties"].(map[string]interface{}); ok {
		for key, value := range properties {
			if prop, ok := value.(map[string]interface{}); ok {
				example[key] = getExampleValue(prop)
			}
		}
	}

	return example
}

// getExampleValue returns an example value based on the property type
func getExampleValue(prop map[string]interface{}) interface{} {
	propType, _ := prop["type"].(string)
	description, _ := prop["description"].(string)

	switch propType {
	case "string":
		if description != "" {
			return "example_" + strings.ToLower(strings.ReplaceAll(description, " ", "_"))[:20]
		}
		return "example_string"
	case "number":
		return 42.5
	case "integer":
		return 42
	case "boolean":
		return true
	case "array":
		if items, ok := prop["items"].(map[string]interface{}); ok {
			if itemType, ok := items["type"].(string); ok {
				switch itemType {
				case "string":
					return []string{"example_item_1", "example_item_2"}
				case "number", "integer":
					return []int{1, 2, 3}
				case "object":
					// Handle array of objects by creating example objects
					exampleObj := getExampleValue(items)
					return []interface{}{exampleObj, exampleObj}
				}
			}
		}
		// Fallback for arrays without proper items definition
		return []interface{}{"item1", "item2"}
	case "object":
		if properties, ok := prop["properties"].(map[string]interface{}); ok {
			obj := make(map[string]interface{})
			for k, v := range properties {
				if subProp, ok := v.(map[string]interface{}); ok {
					obj[k] = getExampleValue(subProp)
				}
			}
			return obj
		}
		return map[string]interface{}{"key": "value"}
	default:
		return "example_value"
	}
}

// extractJSONFromResponse extracts JSON content from a response that may contain markdown or explanatory text
func extractJSONFromResponse(response string) string {
	// First, try to find JSON within markdown code blocks
	jsonStart := strings.Index(response, "```json")
	if jsonStart >= 0 {
		jsonStart += len("```json")
		jsonEnd := strings.Index(response[jsonStart:], "```")
		if jsonEnd > 0 {
			return strings.TrimSpace(response[jsonStart : jsonStart+jsonEnd])
		}
	}

	// Try generic code blocks
	jsonStart = strings.Index(response, "```")
	if jsonStart >= 0 {
		jsonStart += len("```")
		contentAfterMarker := response[jsonStart:]
		newlineIdx := strings.Index(contentAfterMarker, "\n")
		if newlineIdx >= 0 {
			contentAfterMarker = contentAfterMarker[newlineIdx+1:]
		}
		jsonEnd := strings.Index(contentAfterMarker, "```")
		if jsonEnd > 0 {
			extracted := strings.TrimSpace(contentAfterMarker[:jsonEnd])
			if isValidJSONStart(extracted) {
				return extracted
			}
		}
	}

	// Try to find JSON object by looking for { and matching }
	jsonStart = strings.Index(response, "{")
	if jsonStart >= 0 {
		// Find the matching closing brace
		braceCount := 0
		inString := false
		escapeNext := false

		for i := jsonStart; i < len(response); i++ {
			char := response[i]

			if escapeNext {
				escapeNext = false
				continue
			}

			if char == '\\' {
				escapeNext = true
				continue
			}

			if char == '"' {
				inString = !inString
				continue
			}

			if !inString {
				if char == '{' {
					braceCount++
				} else if char == '}' {
					braceCount--
					if braceCount == 0 {
						extracted := strings.TrimSpace(response[jsonStart : i+1])
						if isValidJSONStart(extracted) {
							return extracted
						}
						break
					}
				}
			}
		}
	}

	// If no JSON found, return original response
	return response
}

// isValidJSONStart checks if a string starts with valid JSON
func isValidJSONStart(s string) bool {
	s = strings.TrimSpace(s)
	return strings.HasPrefix(s, "{") || strings.HasPrefix(s, "[")
}

// buildMessagesWithMemory builds Anthropic messages from memory and current prompt
func (c *AnthropicClient) buildMessagesWithMemory(ctx context.Context, prompt string, params *interfaces.GenerateOptions) []Message {
	messages := []Message{}

	// Retrieve and add memory messages if available
	if params.Memory != nil {
		memoryMessages, err := params.Memory.GetMessages(ctx)
		if err != nil {
			c.logger.Error(ctx, "Failed to retrieve memory messages", map[string]interface{}{
				"error": err.Error(),
			})
		} else {
			// Convert memory messages to Anthropic format, ensuring system messages come first
			var systemMessages []Message
			var otherMessages []Message

			for _, msg := range memoryMessages {
				switch msg.Role {
				case interfaces.MessageRoleUser:
					otherMessages = append(otherMessages, Message{
						Role:    "user",
						Content: msg.Content,
					})
				case interfaces.MessageRoleAssistant:
					if msg.Content != "" {
						otherMessages = append(otherMessages, Message{
							Role:    "assistant",
							Content: msg.Content,
						})
					}
				case interfaces.MessageRoleTool:
					// Tool messages in Anthropic are handled as user messages with tool results
					if msg.ToolCallID != "" {
						toolName := "unknown"
						if msg.Metadata != nil {
							if name, ok := msg.Metadata["tool_name"].(string); ok {
								toolName = name
							}
						}
						otherMessages = append(otherMessages, Message{
							Role:    "user",
							Content: fmt.Sprintf("Tool %s result: %s", toolName, msg.Content),
						})
					}
				case interfaces.MessageRoleSystem:
					// System messages in Anthropic are handled separately in the request
					// We collect them here but they won't be added to messages array

					// TODO: Decide how to handle system messages in Anthropic
					// systemMessages = append(systemMessages, Message{
					// 	Role:    "system",
					// 	Content: msg.Content,
					// })
				}
			}

			// Add system messages first, then other messages
			messages = append(messages, systemMessages...)
			messages = append(messages, otherMessages...)
		}
	}

	// Add current user message
	messages = append(messages, Message{
		Role:    "user",
		Content: prompt,
	})

	return messages
}
